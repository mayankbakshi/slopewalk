"""The extra panels: each case's slope, R over one or two weights, R along the step, the hidden layer, momentum;
and the changes to the existing panels: frozen weights, predictions in the history, the start boundary, log scale."""

import matplotlib.pyplot as plt
import numpy as np
import pytest
import torch
import torch.nn as nn
from helpers import make
from render_examples import reference_net

from slopewalk import Session, explore, xor_centres
from slopewalk.draw import THEME
from slopewalk.picture import Picture, bottom_panels

XOR = xor_centres()
CLUSTERS = xor_centres(noise=0.35)


@pytest.fixture
def static(monkeypatch):
    monkeypatch.setenv("NB_STATIC", "1")
    monkeypatch.setattr(plt, "show", lambda *a, **k: None)
    yield
    plt.close("all")


def axes_titled(fig, start):
    return [ax for ax in fig.axes if ax.get_title().startswith(start)]


def test_views_add_bottom_panels_in_rows_of_three(static):
    s = explore(make("tanh3"), *XOR, views=("momentum", "line", "slice", "hidden", "cases"))
    assert bottom_panels(s) == ["plane", "history", "table", "cases", "slice", "line", "hidden", "momentum"]
    assert len(s.fig.axes) == 4 + 8
    plain = explore(make("tanh3"), *XOR)
    assert len(plain.fig.axes) == 4 + 3
    assert s.fig.get_size_inches()[1] == pytest.approx(plain.fig.get_size_inches()[1] + 2 * 4.2)
    assert all(not ax.get_title().startswith(("A: ", "B: ")) for ax in s.fig.axes[4:])


def test_case_slopes_panel(static):
    s = explore(reference_net(40), *CLUSTERS, case_slopes="w10")
    assert s.views == ("cases",) and s.case_weight == 9
    ax = axes_titled(s.fig, "each case's slope along w10")[0]
    dots = np.concatenate([c.get_offsets()[:, 1] for c in ax.collections if len(c.get_offsets()) > 1])
    assert len(dots) == 200
    average = [line for line in ax.lines if line.get_linestyle() == "--"][0].get_ydata()[0]
    assert average == pytest.approx(s.slopes()[9], abs=1e-6)  # the dashed line is dR/dw10
    assert average == pytest.approx(dots.mean(), abs=1e-6)


def test_slice_over_two_weights(static):
    s = explore(reference_net(40), *XOR, eta=0.5, params=["w10", "w11"], views="slice")
    assert s.slice_weights == [9, 10]
    s.step(4)
    fig = s.draw()
    ax = axes_titled(fig, "R over w10 and w11")[0]
    assert ax.get_xlabel() == "w10" and ax.get_ylabel() == "w11"
    path = [line for line in ax.lines if line.get_label() == "steps taken"][0]
    assert len(path.get_xdata()) == 5 and path.get_xdata()[-1] == pytest.approx(s.weights()[9])
    assert "other weights" not in ax.get_title()  # only these two weights move


def test_slice_grid_matches_risk(static):
    s = Session(reference_net(40), *XOR)
    w = np.array(s.weights())
    W = np.repeat(w[None, :], 3, axis=0)
    W[:, 9] = [0.0, 1.0, 2.0]
    assert s.model.risk_batch(s.data.X, s.data.y, W) == pytest.approx([s.risk(list(row)) for row in W], abs=1e-6)


def test_slice_over_one_weight_and_defaults(static):
    s = explore(make("tanh3"), *XOR, slice_weights="w12")
    ax = axes_titled(s.fig, "R over w12")[0]
    assert ax.get_xlabel() == "w12" and any(line.get_label().startswith("tangent") for line in ax.lines)
    assert Session(make("tanh3"), *XOR).slice_weights == [9, 10]  # the first two output weights
    assert Session(make("tanh3"), *XOR, params=["w1"]).slice_weights == [0]


def test_line_view(static):
    s = explore(reference_net(40), *XOR, eta=0.5, views="line")
    ax = axes_titled(s.fig, "R along the next step")[0]
    curve = [line for line in ax.lines if line.get_label() == "R along the step"][0]
    tangent = [line for line in ax.lines if line.get_label() == "first-order prediction"][0]
    g = s.slopes()
    assert curve.get_ydata()[0] == pytest.approx(s.risk(), abs=1e-6)
    t, R = curve.get_xdata(), curve.get_ydata()
    assert tangent.get_ydata() == pytest.approx(s.risk() - t * float((g**2).sum()), abs=1e-9)
    w = np.array(s.weights())
    assert R[20] == pytest.approx(s.risk(list(w - t[20] * g)), abs=1e-6)


def test_hidden_view_for_one_to_four_units(static):
    shapes = {}
    for m in (1, 2, 3, 4):
        torch.manual_seed(0)
        net = nn.Sequential(nn.Linear(2, m), nn.Tanh(), nn.Linear(m, 1))
        s = explore(net, *XOR, views="hidden")
        ax = axes_titled(s.fig, "the cases at their hidden values")[0]
        shapes[m] = hasattr(ax, "set_zlim")
        plt.close("all")
    assert shapes == {1: False, 2: False, 3: True, 4: False}  # 3-D only for three units
    s = explore(make("logistic"), *XOR, views="hidden")
    assert any("no hidden layer" in t.get_text() for ax in s.fig.axes for t in ax.texts)


def test_momentum_view(static):
    s = explore(reference_net(40), *XOR, eta=0.5, views=("momentum", "slice"), lookahead=12, beta=0.7)
    ax = axes_titled(s.fig, "the next 12 steps")[0]
    gd, mo = ax.lines[0].get_ydata(), ax.lines[1].get_ydata()
    look = s.lookahead()
    assert list(gd) == pytest.approx(look["gd"]["R"]) and list(mo) == pytest.approx(look["momentum"]["R"])
    slice_ax = axes_titled(s.fig, "R over w10 and w11")[0]
    labels = [line.get_label() for line in slice_ax.lines]
    assert "next steps, gradient descent" in labels and "next steps, momentum 0.7" in labels


def test_frozen_weights_are_drawn_lighter(static):
    s = explore(make("tanh3"), *XOR, params="output")
    ax = s.fig.axes[0]
    lines = [line for line in ax.lines if line.get_zorder() == 1]
    alphas = [line.get_alpha() for line in lines]
    assert alphas[:9] == [THEME.frozen_alpha] * 9 and alphas[9:] == [1.0] * 4
    table = [tbl for a in s.fig.axes for tbl in a.tables][0]
    cells = table.get_celld()
    assert cells[1, 0].get_text().get_color() == THEME.frozen_text  # w1 does not move
    assert cells[10, 0].get_text().get_color() != THEME.frozen_text  # w10 moves


def test_history_shows_predictions_and_notes(static):
    s = explore(reference_net(40), *XOR, eta=0.5)
    s.step(3)
    ax = axes_titled(s.draw(), "R = ")[0]
    assert len(ax.lines[0].get_xdata()) == 4  # R after each step
    hollow = ax.lines[1]
    assert list(hollow.get_ydata()) == pytest.approx([p.predicted for p in s.predictions])
    note = [t.get_text() for t in ax.texts][0]
    last = s.predictions[-1]
    assert f"last step: predicted {last.predicted - last.before:+.4f}, actual {last.actual - last.before:+.4f}" in note
    assert "next step, eta 0.5: predicted change" in note


def test_log_scale(static):
    s = explore(reference_net(40), *XOR, eta=0.5, log_scale=True)
    s.step(2)
    assert axes_titled(s.draw(), "R = ")[0].get_yscale() == "log"


def test_start_boundary_overlay(static):
    s = explore(reference_net(40), *XOR, eta=0.5)
    plane = axes_titled(s.fig, "p for class 1")[0]
    dashed_before = len(plane.collections)
    s.step(20)
    plane = axes_titled(s.draw(), "p for class 1")[0]
    assert len(plane.collections) > dashed_before  # the dashed starting boundary appears once the weights moved


def test_picture_keeps_its_axes_when_refreshed():
    s = Session(make("tanh3"), *XOR, views=("line",))
    pic = Picture(s)
    pic.refresh()
    positions = [tuple(ax.get_position().bounds) for ax in pic.fig.axes]
    png = pic.png()
    assert png[:8] == b"\x89PNG\r\n\x1a\n"
    s.step(3)
    pic.refresh()
    assert [tuple(ax.get_position().bounds) for ax in pic.fig.axes] == positions
    assert plt.get_fignums() == []  # never a pyplot figure
