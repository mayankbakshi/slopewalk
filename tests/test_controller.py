"""The widgets, built without a front end: sliders and buttons change the network, the history and the picture."""

import copy
import re

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
    assert list(C.buttons) == ["gradient step", "10 steps", "back one step", "reset"]
    assert C.tabs is None  # no tabs unless the call asks for the backward pass or for extra views
    hint, row, panel, frozen = C.controls.children
    assert "Press gradient step" in hint.value and panel is C.panel and frozen is C.frozen_note
    assert list(row.children) == [C.eta, C.button_row]  # eta, then the step buttons as one group
    b = C.buttons
    assert list(C.button_row.children) == [b["gradient step"], C.count, b["10 steps"], b["back one step"], b["reset"]]
    assert (
        C.button_row.layout.flex_flow == "row nowrap"
        and C.count.value == 10
        and b["10 steps"].description == "steps"  # joined to the box: "10 | steps"
    )
    assert C.layout_pick in C.layout_row.children
    assert re.sub("<[^>]+>", "", C.message.value) == ""  # nothing to report before the first action
    assert "next step, eta 0.7: predicted change" in C.readout.value and "none yet" in C.readout.value
    assert isinstance(C.image, ipywidgets.Image) and C.image.value[:8] == b"\x89PNG\r\n\x1a\n"
    (root,) = C.widgets  # the picture first, then the status, the controls, and last the layout of the picture
    assert list(root.children) == [C.style, C.image, C.status, C.controls, C.layout_row, C.out]
    assert list(C.status.children) == [C.readout, C.message]
    assert C.image.layout.width == "100%"  # the default layout outside Colab is wide: the picture at the full width
    for html in (C.message.value, C.readout.value, hint.value):
        assert "border-left" not in html and "border-radius" not in html and "Try this" not in html
    assert s.fig is not None and s.fig is C.picture.fig  # drawn once at the start, off pyplot
    assert s.fig not in [plt.figure(n) for n in plt.get_fignums()]
    tooltip = getattr(C.buttons["gradient step"], "tooltip", None)
    assert tooltip and "eta" in tooltip


def test_standard_layout_has_the_same_order(live):
    net, X, y = logistic_start()
    C = explore(net, X, y, layout="standard").controller
    (root,) = C.widgets
    assert list(root.children) == [C.style, C.image, C.status, C.controls, C.layout_row, C.out]
    assert C.image.layout.width == ""  # the picture keeps its own width


def test_layout_switch(live):
    s = explore(make("tanh3"), *xor_centres())
    C = s.controller
    assert C.layout_pick.value == "wide" and s.fig.get_size_inches()[0] == 27
    C.layout_pick.value = "standard"
    assert s.layout.style == "standard" and s.fig.get_size_inches()[0] == 15
    order = [C.style, C.image, C.status, C.controls, C.layout_row, C.out]
    assert list(C.root.children) == order and "Layout standard" in C.message.value
    assert C.image.layout.width == ""
    C.layout_pick.value = "wide"
    assert s.fig.get_size_inches()[0] == 27 and C.image.layout.width == "100%"
    assert list(C.root.children) == order


def test_messages_say_what_happened(live):
    net, X, y = logistic_start()
    s = explore(net, X, y)
    C = s.controller
    C.sliders[0].value = 0.6
    assert "w1 changed from 0.50 to 0.60: R went from 1.0032 to" in C.message.value
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
    labels = [t.get_text() for ax in s.fig.axes for t in ax.texts if t.get_fontsize() == s.theme.edge_label_size]
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
    assert len(hist_ax.lines[0].get_xdata()) == 11 and hist_ax.get_title() == f"R = {s.history[-1]:.4f}"


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
    assert "next step, eta 0.5: predicted change" in C.readout.value and "none yet" in C.readout.value
    columns = C.readout.value.count("<td")
    C.buttons["gradient step"].click()
    p = s.predictions[-1]
    assert f"{p.actual - p.before:+.4f}" in C.readout.value
    assert f"predicted {p.predicted - p.before:+.4f}" in C.readout.value
    assert "#2E8B57" in C.readout.value and "none yet" not in C.readout.value  # R went down: green
    assert C.readout.value.count("<td") == columns  # the status keeps its shape after the first step
    C.eta.value = 2.0
    assert s.eta == 2.0 and "next step, eta 2: predicted change" in C.readout.value


def test_tabs_appear_only_when_asked_for(live):
    X, y, groups = xor_centres()
    assert explore(make("tanh3"), X, y, groups).controller.tabs is None
    titles = lambda C: [C.tabs.get_title(i) for i in range(len(C.tabs.children))]  # noqa: E731
    C = explore(make("tanh3"), X, y, groups, views="backprop").controller
    assert titles(C) == ["Weights and steps", "Backpropagation"] and C.tabs.selected_index == 1
    C = explore(make("tanh3"), X, y, groups, views="slice").controller
    assert titles(C) == ["Weights and steps", "Views"] and C.tabs.selected_index == 0
    C = explore(make("tanh3"), X, y, groups, views=("backprop", "line"), case_slopes="w10").controller
    assert titles(C) == ["Weights and steps", "Backpropagation", "Views"]
    assert list(C.root.children) == [C.style, C.image, C.status, C.tabs, C.layout_row, C.out]  # tabs under the picture
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


def test_changing_tabs_leaves_the_picture_as_it_is(live):
    s = explore(make("tanh3"), *xor_centres(), views=("backprop", "slice"), backprop_step=2)
    C = s.controller
    fig, png, views = s.fig, C.image.value, s.views
    for tab in (0, 2, 1, 0):
        C.tabs.selected_index = tab
        assert s.fig is fig and C.image.value == png and s.views == views and s.backprop_step == 2


def test_back_one_step(live):
    s = explore(make("tanh3"), *xor_centres(), eta=0.5)
    C = s.controller
    back = C.buttons["back one step"]
    assert back.disabled  # nothing to undo yet
    w0 = s.weights()
    C.buttons["gradient step"].click()
    assert not back.disabled and len(s.history) == 2
    R1 = s.history[-1]
    back.click()
    assert s.weights() == pytest.approx(w0) and C.values() == pytest.approx(w0) and s.history == [s.risk(w0)]
    assert f"Back one step: R went from {R1:.4f} to {s.history[0]:.4f}" in C.message.value and back.disabled


def test_buttons_are_disabled_while_working(live, monkeypatch):
    s = explore(make("tanh3"), *xor_centres(), eta=0.5)
    C = s.controller
    seen = []
    C.buttons["gradient step"].observe(lambda ch: seen.append(ch["new"]), names="disabled")
    C.buttons["gradient step"].click()
    assert seen == [True, False]  # disabled for the step and the redraw, then enabled again

    def broken(*a, **k):
        raise RuntimeError("drawing failed")

    monkeypatch.setattr(C.picture, "refresh", broken)
    C.buttons["10 steps"].click()  # the callback fails; ipywidgets logs the error
    assert not any(b.disabled for name, b in C.buttons.items() if name != "back one step")


def test_text_takes_the_colour_of_the_root(live):
    """No fixed text colours in the control area: text takes the colour of the root box, set by the theme; only the
    change of the last step is coloured, in colours that keep 3:1 on light and dark backgrounds."""
    s = explore(make("tanh3"), *xor_centres(), views=("backprop", "slice"))
    C = s.controller
    C.buttons["gradient step"].click()
    html = [C.message.value, C.readout.value, C.layout_row.children[0].value]
    html += [w.value for tab in C.tabs.children for w in tab.children if isinstance(w, ipywidgets.HTML)]
    colours = {c for h in html for c in re.findall(r"color:\s*(#[0-9A-Fa-f]{3,6})", h)}
    assert colours <= {"#2E8B57", "#E53935"}


def test_sliders_apply_on_release(live):
    s = explore(make("tanh3"), *xor_centres(), views=("backprop", "momentum"))
    C = s.controller
    assert not any(w.continuous_update for w in [*C.sliders, C.eta, C.beta, C.ahead])


def test_a_slider_marks_its_weight(live):
    s = explore(make("tanh3"), *xor_centres())
    C = s.controller
    C.sliders[4].value = 0.25
    assert s.last_moved == 4
    labels = [t.get_text() for t in s.fig.axes[0].texts if t.get_fontsize() == s.theme.edge_label_size]
    assert "w5 = 0.25" in labels
    C.buttons["gradient step"].click()
    assert s.last_moved is None


def test_the_change_is_the_last_action(live):
    """The large change in the status follows the last action: one step, ten steps, a slider move, back one step,
    and reset; changing eta leaves it as it is."""
    s = explore(make("tanh3"), *xor_centres(), eta=0.5)
    C = s.controller
    numbers = lambda: re.sub("<[^>]+>", " ", C.readout.value)  # noqa: E731
    C.buttons["gradient step"].click()
    p = s.predictions[-1]
    assert C.action == pytest.approx((p.actual - p.before, C.action[1])) and C.action[1].startswith("one step")
    R1 = s.history[-1]
    C.buttons["10 steps"].click()
    total = s.history[-1] - R1
    predicted = sum(q.predicted - q.before for q in s.predictions[-10:])
    assert C.action[0] == pytest.approx(total, abs=1e-6) and f"{total:+.4f}" in numbers()
    assert C.action[1] == f"10 steps, predicted {predicted:+.4f}"
    before = s.history[-1]
    C.sliders[4].value += 0.3
    assert C.action[1] == "w5 moved by hand" and C.action[0] == pytest.approx(s.risk() - before, abs=1e-6)
    C.eta.value = 1.0
    assert C.action[1] == "w5 moved by hand"  # eta does not change R
    R_moved = s.risk()
    C.buttons["back one step"].click()
    assert C.action[1] == "back one step" and C.action[0] == pytest.approx(s.risk() - R_moved, abs=1e-6)
    C.buttons["reset"].click()
    assert C.action is None and "none yet" in C.readout.value
    s.step(3)  # steps in code count as an action too
    assert C.action[1].startswith("3 steps, predicted")


def test_view_settings_follow_their_view(live):
    s = explore(make("tanh3"), *xor_centres(), views="slice")
    C = s.controller
    assert all(not w.disabled for w in C.slice_pick) and C.case_pick.disabled and C.beta.disabled and C.ahead.disabled
    C.view_boxes["momentum"].value = True
    assert not C.beta.disabled and not C.ahead.disabled
    C.view_boxes["slice"].value = False
    assert all(w.disabled for w in C.slice_pick)
    tab = C.tabs.children[C.kinds.index("views")]
    slice_row = [row for row in tab.children if C.view_boxes["slice"] in getattr(row, "children", ())][0]
    assert all(w in slice_row.children for w in C.slice_pick)  # the settings sit beside their checkbox


def test_backprop_wording(live):
    s = explore(make("tanh3"), *xor_centres(), views="backprop")
    C = s.controller
    hint = C.tabs.children[1].children[0].value
    assert "Worked example below the networks" in hint and "under the picture" not in hint
    assert C.tabs.get_title(1) == "Backpropagation" and C.bp_case.description == "Worked example"


def test_controls_follow_the_colours_switch(live):
    """The root box carries the class of the picture's colours, and the style sets the colours of ipywidgets for both
    themes from the picture's palette; the switch changes the class."""
    from slopewalk.draw import DARK_THEME, THEME
    from slopewalk.widgets import control_css

    s = explore(make("tanh3"), *xor_centres())
    C = s.controller
    assert "slopewalk-light" in C.root._dom_classes and "slopewalk-dark" not in C.root._dom_classes
    assert C.style in C.root.children and C.style.layout.display == "none" and C.style.value == control_css()
    css = control_css()
    for name, T in (("light", THEME), ("dark", DARK_THEME)):
        rule = css[css.index(f".slopewalk-{name}{{") :]
        rule = rule[: rule.index("}")]
        assert f"background-color:{T.paper}" in rule and f"--jp-widgets-color:{T.ink}" in rule
        assert f"--jp-layout-color1:{T.paper}" in rule and f"color-scheme:{name}" in rule
        for var in ("--jp-ui-font-color1", "--jp-widgets-label-color", "--jp-widgets-readout-color"):
            assert f"{var}:{T.ink}" in rule
    C.theme_pick.value = "dark"
    assert "slopewalk-dark" in C.root._dom_classes and "slopewalk-light" not in C.root._dom_classes
    assert s.theme_name == "dark"
    C.theme_pick.value = "light"
    assert C.root._dom_classes.count("slopewalk-light") == 1 and "slopewalk-dark" not in C.root._dom_classes
    dark = explore(make("tanh3"), *xor_centres(), theme="dark").controller
    assert "slopewalk-dark" in dark.root._dom_classes


def test_a_chosen_number_of_steps(live):
    """The box joined to the steps button sets how many steps it takes; the button says "steps" ("step" for 1), so
    that box and button read as one control."""
    s = explore(make("tanh3"), *xor_centres(), eta=0.5)
    C = s.controller
    button = C.buttons["10 steps"]
    C.count.value = 25
    assert button.description == "steps" and "25 gradient steps" in button.tooltip
    assert C.count.layout.margin.split()[1] == "0" and button.layout.margin.split()[3] == "0"  # no gap between them
    R0 = s.history[-1]
    button.click()
    assert len(s.history) == 26 and s.history[-1] < R0
    assert C.action[1].startswith("25 steps, predicted") and "25 gradient steps with eta 0.5" in C.message.value
    C.count.value = 1
    assert button.description == "step"
    button.click()
    assert len(s.history) == 27 and C.action[1].startswith("one step")
    assert (C.count.min, C.count.max) == (1, 1000)


def test_empty_message_leaves_no_gap_and_the_stage_is_named(live):
    s = explore(make("tanh3"), *xor_centres(), views="backprop")
    C = s.controller
    assert "Press next stage" in C.message.value and C.message.layout.display == ""
    C.say("")
    assert C.message.layout.display == "none" and C.message.value == ""
    assert "stage 0 of 6" in C.bp_where.value
    C.bp_next.click()
    assert "stage 1 of 6" in C.bp_where.value
    row = C.tabs.children[C.kinds.index("backprop")].children[1]
    assert list(row.children)[:3] == [C.bp_prev, C.bp_where, C.bp_next]
    plain = explore(make("tanh3"), *xor_centres()).controller
    assert plain.message.layout.display == "none"  # nothing to say before the first action
