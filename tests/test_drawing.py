"""Drawing: still pictures across architectures, the panel grid, the colours and widths of the lines, save()."""

import os
import pathlib
import re

import matplotlib.pyplot as plt
import pytest
import torch
from helpers import NETS, make, points

from slopewalk import Session, explore, two_points, xor_centres
from slopewalk.draw import THEME


@pytest.fixture
def static(monkeypatch):
    monkeypatch.setenv("NB_STATIC", "1")
    monkeypatch.setattr(plt, "show", lambda *a, **k: None)
    yield
    plt.close("all")


def panels(fig):
    return [ax for ax in fig.axes if re.match(r"^[A-Z]: ", ax.get_title())]


def weight_lines(ax):
    return [line for line in ax.lines if line.get_zorder() == 1]


def edge_texts(ax):
    return [t for t in ax.texts if t.get_fontsize() == THEME.edge_label_size]


def logistic_start():
    net = make("logistic")
    with torch.no_grad():
        net[0].weight[:] = torch.tensor([[0.5, 0.5]])
        net[0].bias[:] = 0.0
    X, y, _ = two_points()
    return net, X, y


@pytest.mark.parametrize("name", list(NETS))
def test_still_picture_for_every_architecture(name, static, capsys):
    net = make(name)
    X, y = points(net, n=3)
    s = explore(net, X, y)
    assert isinstance(s, Session) and s.controller is None and s.fig is plt.gcf()
    assert "Still picture of the start" in capsys.readouterr().out
    assert len(panels(s.fig)) == 3 and len(s.fig.axes) == 3 + 3
    for ax in panels(s.fig):
        assert len(weight_lines(ax)) == len(s.names)
        assert len(edge_texts(ax)) in (0, len(s.names))  # all labels, or none when they cannot be placed
    assert s.history == [s.risk(s.w0)] and s.weights() == s.w0


def test_four_centres_form_the_2x2_grid_in_order(static):
    X, y, groups = xor_centres()
    s = explore(make("tanh3"), X, y, groups)
    P = panels(s.fig)
    assert [ax.get_title()[0] for ax in P] == ["A", "B", "C", "D"]
    pos = [ax.get_position() for ax in P]
    assert pos[0].x0 < pos[1].x0 and pos[0].y0 == pytest.approx(pos[1].y0)  # A left of B
    assert pos[2].x0 < pos[3].x0 and pos[2].y0 == pytest.approx(pos[3].y0)  # C left of D
    assert pos[0].y0 > pos[2].y0 and pos[0].x0 == pytest.approx(pos[2].x0)  # A above C


def test_deep_network_uses_one_panel_per_row(static):
    X, y, groups = xor_centres()
    s = explore(make("two_hidden"), X, y, groups)
    pos = [ax.get_position() for ax in panels(s.fig)]
    assert all(p.x0 == pytest.approx(pos[0].x0) for p in pos)
    assert all(pos[i].y0 > pos[i + 1].y0 for i in range(3))


def test_line_colours_widths_and_labels(static):
    net, X, y = logistic_start()
    with torch.no_grad():
        net[0].weight[0, 1] = -0.5
    s = explore(net, X, y)
    ax = panels(s.fig)[0]
    lines, labels = weight_lines(ax), edge_texts(ax)
    assert [line.get_color() for line in lines] == [THEME.positive, THEME.negative, THEME.positive]
    width = lambda v: THEME.line_width + THEME.line_slope * abs(v)  # noqa: E731
    assert [line.get_linewidth() for line in lines] == pytest.approx([width(0.5), width(0.5), width(0.0)])
    assert [t.get_text() for t in labels] == ["w1 = 0.50", "w2 = -0.50", "w3 = 0.00"]
    assert [t.get_color() for t in labels] == [THEME.positive, THEME.negative_text, THEME.positive]
    assert all(t.get_fontweight() == "normal" for t in labels)
    s.draw([0.6, -0.5, 0.0])  # a weight moved by hand: its sign colour, and a bold label
    ax = panels(s.fig)[0]
    lines, labels = weight_lines(ax), edge_texts(ax)
    assert lines[0].get_color() == THEME.positive and lines[1].get_color() == THEME.negative
    assert labels[0].get_text() == "w1 = 0.60" and labels[0].get_color() == THEME.positive
    assert labels[0].get_fontweight() == "bold" and labels[1].get_fontweight() == "normal"
    assert lines[0].get_linewidth() == pytest.approx(width(0.6))


def test_mixed_group_is_drawn_black(static):
    net, X, y = logistic_start()
    s = explore(net, X, y, groups=[0, 0])
    ax = panels(s.fig)[0]
    assert ax.get_title().startswith("A: mean of 2 points (0.0, 1.0), classes mixed:")
    assert ax.title.get_color() == THEME.mixed_colour
    assert any(t.get_text().startswith("y mixed\nloss ") for t in ax.texts)


def test_history_steps_and_reset_on_the_session(static):
    net, X, y = logistic_start()
    s = explore(net, X, y)
    R0 = s.history[0]
    w = s.step(3, eta=1.0)
    assert len(s.history) == 4 and s.history[-1] < R0 and s.weights() == pytest.approx(w)
    fig = s.draw()
    hist_ax = [ax for ax in fig.axes if ax.get_title().startswith("R = ")][0]
    assert hist_ax.get_title() == f"R = {s.history[-1]:.4f}" and len(hist_ax.lines[0].get_xdata()) == 4
    assert s.reset() == s.w0 and s.weights() == s.w0 and s.history == [R0]


def test_save_writes_the_picture_and_closes_its_figure(static, tmp_path):
    net, X, y = logistic_start()
    s = explore(net, X, y)
    open_before = plt.get_fignums()
    png = s.save(tmp_path / "state.png")
    assert os.path.getsize(png) > 10_000 and plt.get_fignums() == open_before
    s.step(2)
    png2 = s.save(tmp_path / "after.png")
    assert pathlib.Path(png).read_bytes() != pathlib.Path(png2).read_bytes()
    assert plt.get_fignums() == open_before
    assert os.path.getsize(s.save(tmp_path / "state.pdf")) > 1000


def test_session_repr_and_silent_display(static):
    net, X, y = logistic_start()
    s = explore(net, X, y)
    assert repr(s) == "<slopewalk session: 3 weights, R = 1.003>"
    assert s._ipython_display_() is None


def test_weight_labels_in_the_first_panel_only(static):
    """The weights are the same in every panel, so only panel A labels them; lines keep their width and colour."""
    s = explore(make("tanh3"), *xor_centres())
    P = panels(s.fig)
    assert len(edge_texts(P[0])) == 13 and all(not edge_texts(ax) for ax in P[1:])
    assert all(len(weight_lines(ax)) == 13 for ax in P)


def test_the_weight_moved_last_keeps_its_name(static):
    s = explore(make("tanh3"), *xor_centres())  # 13 weights: values only on the lines
    w = s.weights()
    w[4] += 0.3
    s.model.set_weights(w)
    s.last_moved = 4
    fig = s.draw()
    texts = [t.get_text() for t in edge_texts(panels(fig)[0])]
    assert f"w5 = {w[4]:.2f}" in texts and sum(" = " in t for t in texts) == 1 and len(texts) == 13
    s.step(1)  # a step moves every weight: no name any more
    assert s.last_moved is None
    texts = [t.get_text() for t in edge_texts(panels(s.draw())[0])]
    assert not any(" = " in t for t in texts)
    many = explore(make("wide"), *xor_centres())  # 49 weights: the name must not cost the other labels
    many.last_moved = 0
    texts = [t.get_text() for t in edge_texts(panels(many.draw())[0])]
    assert len(texts) == 49


def test_colours_and_type_scale(static):
    """Class 1 in a red that reads as text, negative lines in an orange with 3:1 on white, one size for the titles
    of the panels below the networks."""
    assert THEME.class_colour == {0: "#0969DA", 1: "#CF222E", 2: "#00796B", 3: "#B4237F"}
    assert THEME.negative == "#D1600A"
    s = explore(make("tanh3"), *xor_centres(), views=("cases", "slice", "line", "hidden", "momentum", "backprop"))
    lower = s.fig.axes[4:]
    sizes = {ax.title.get_fontsize() for ax in lower if ax.get_title()} | {
        ax._left_title.get_fontsize() for ax in lower if ax.get_title(loc="left")
    }
    assert sizes == {THEME.panel_title_size}
    legends = {t.get_fontsize() for ax in lower if ax.get_legend() for t in ax.get_legend().get_texts()}
    assert legends == {THEME.legend_size}


def test_lines_keep_their_sign_and_moved_weights_are_bold(static):
    """Lines always show the sign of their weight; blue is kept for class 0. The weight moved by hand since the last
    step has a bold label in panel A, with its name; a step clears it, step_back brings it back."""
    s = explore(make("tanh3"), *xor_centres(), eta=0.5)
    sign = lambda w: [THEME.positive if v >= 0 else THEME.negative for v in w]  # noqa: E731
    colours = lambda fig: [[line.get_color() for line in weight_lines(ax)] for ax in panels(fig)]  # noqa: E731
    bold = lambda fig: [t.get_text() for t in edge_texts(panels(fig)[0]) if t.get_fontweight() == "bold"]  # noqa: E731
    s.step(1)
    fig = s.draw()
    assert all(panel == sign(s.weights()) for panel in colours(fig)) and bold(fig) == []
    w = s.weights()
    w[4] += 0.3  # what a slider does
    s.model.set_weights(w)
    s.last_moved = 4
    fig = s.draw()
    assert all(panel == sign(w) for panel in colours(fig)) and bold(fig) == [f"w5 = {w[4]:.2f}"]
    s.step(1)
    assert bold(s.draw()) == []
    s.step_back()  # back to the weights with w5 moved by hand
    assert len(bold(s.draw())) == 1


def test_note_in_the_panels_without_numbers(static):
    note = lambda ax: [t for t in ax.texts if t.get_text() == "weights as in A"]  # noqa: E731
    s = explore(make("tanh3"), *xor_centres())
    P = panels(s.fig)
    assert not note(P[0]) and all(len(note(ax)) == 1 for ax in P[1:])
    torch.manual_seed(0)
    net = torch.nn.Sequential(torch.nn.Linear(2, 20), torch.nn.Tanh(), torch.nn.Linear(20, 1))  # 81 weights
    crowded = explore(net, *xor_centres())
    assert not any(note(ax) for ax in panels(crowded.fig))  # panel A has no numbers either: no note


def test_r_axis_shows_progress(static):
    s = explore(make("tanh3"), *xor_centres(), eta=0.5)
    s.step(10)
    ax = [a for a in s.draw().axes if a.get_title().startswith("R = ")][0]
    lo, hi = ax.get_ylim()
    values = list(s.history) + [p.predicted for p in s.predictions]
    assert lo >= 0.0 and (max(values) - min(values)) >= 0.5 * (hi - lo)


def test_group_letters_have_a_halo(static):
    s = explore(make("tanh3"), *xor_centres(noise=0.35))
    plane = [a for a in s.fig.axes if a.get_title().startswith("p for class 1")][0]
    letters = [t for t in plane.texts if t.get_text().strip() in "ABCD" and t.get_text().strip()]
    assert len(letters) == 4 and all(t.get_path_effects() for t in letters)
