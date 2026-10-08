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
    assert [line.get_linewidth() for line in lines] == pytest.approx([0.6 + 1.3 * 0.5, 0.6 + 1.3 * 0.5, 0.6])
    assert [t.get_text() for t in labels] == ["w1 = 0.50", "w2 = -0.50", "w3 = 0.00"]
    assert [t.get_color() for t in labels] == [THEME.positive, THEME.negative_text, THEME.positive]
    assert all(t.get_fontweight() == "normal" for t in labels)
    s.draw([0.6, -0.5, 0.0])  # a changed weight: blue and bold
    ax = panels(s.fig)[0]
    lines, labels = weight_lines(ax), edge_texts(ax)
    assert lines[0].get_color() == THEME.changed and lines[1].get_color() == THEME.negative
    assert labels[0].get_text() == "w1 = 0.60" and labels[0].get_color() == THEME.changed
    assert labels[0].get_fontweight() == "bold" and labels[1].get_fontweight() == "normal"
    assert lines[0].get_linewidth() == pytest.approx(0.6 + 1.3 * 0.6)


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
    assert hist_ax.get_title() == f"R = {s.history[-1]:.3f}" and len(hist_ax.lines[0].get_xdata()) == 4
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
