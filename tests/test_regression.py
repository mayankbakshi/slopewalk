"""task="regression": R is the mean squared error; the panels show the prediction and the target."""

import copy
import re

import matplotlib.pyplot as plt
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from helpers import NETS, make, naming_order, regression_points, sine_data

from slopewalk import explore
from slopewalk.model import Model


@pytest.fixture
def static(monkeypatch):
    monkeypatch.setenv("NB_STATIC", "1")
    monkeypatch.setattr(plt, "show", lambda *a, **k: None)
    yield
    plt.close("all")


def texts(fig):
    return [t.get_text() for ax in fig.axes for t in ax.texts]


def panels(fig):
    return [ax for ax in fig.axes if re.match(r"^[A-Z]: ", ax.get_title())]


@pytest.mark.parametrize("name", [n for n in NETS if n != "wide"])
def test_risk_and_slopes(name):
    net = make(name)
    X, y = regression_points(net)
    M = Model(net, task="regression")
    with torch.no_grad():
        expected = F.mse_loss(net(X).reshape(-1), y).item()
    assert M.risk(X, y) == expected
    ref = copy.deepcopy(net)
    F.mse_loss(ref(X).reshape(-1), y).backward()
    grads = [g for m in ref.modules() if isinstance(m, nn.Linear) for g in (m.weight.grad, m.bias.grad)]
    flat = []
    for gw, gb in zip(grads[0::2], grads[1::2]):
        for j in range(gw.shape[0]):
            flat += gw[j].tolist() + [gb[j].item()]
    assert M.slopes(X, y).tolist() == pytest.approx(flat, abs=1e-7)
    # central differences in float64
    M64 = Model(copy.deepcopy(net).double(), task="regression")
    w = M64.get_weights()
    h = 1e-5
    fd = []
    for k in range(len(w)):
        up, down = list(w), list(w)
        up[k] += h
        down[k] -= h
        fd.append((M64.risk(X.double(), y.double(), up) - M64.risk(X.double(), y.double(), down)) / (2 * h))
    assert M.slopes(X, y).tolist() == pytest.approx(fd, abs=1e-4)
    out = M.forward(X, y)
    assert out.p is None and out.loss.tolist() == pytest.approx(((out.a - y.numpy()) ** 2).tolist(), abs=1e-6)


def test_one_input_makes_four_groups_along_x(static):
    """Regression with one input and more than 8 points: four groups of neighbouring points along x1, in order."""
    X, y, _ = sine_data()
    s = explore(sine_net(), X.flip(0), y.flip(0), task="regression")  # the order of the points does not matter
    order = s.data.X[:, 0].argsort()
    assert s.data.groups[order].tolist() == [0] * 4 + [1] * 4 + [2] * 4 + [3] * 4


def sine_net():
    torch.manual_seed(0)
    return nn.Sequential(nn.Linear(1, 3), nn.Tanh(), nn.Linear(3, 1))


def test_regression_picture(static):
    X, y, groups = sine_data()
    s = explore(sine_net(), X, y, groups, eta=0.2, task="regression")
    fig = s.fig
    P = panels(fig)
    assert len(P) == 4 and len(fig.axes) == 4 + 3
    assert P[0].get_title().startswith("A: mean of 4 points (-1.6), y = ")
    assert P[0].title.get_color() == s.theme.neutral_colour
    sigmoid_glyphs = [t for t in texts(fig) if t == "sigmoid"]
    assert sigmoid_glyphs == []  # no sigmoid and no p box
    assert any(t.startswith("ŷ\n") for t in texts(fig)) and not any(t.startswith("p\n") for t in texts(fig))
    assert any(t.startswith("y ") and "\nloss " in t and "change\nfrom start\n+0.000" in t for t in texts(fig))
    plane = [ax for ax in fig.axes if ax.get_title() == "prediction and the points"]
    assert len(plane) == 1
    history = [ax for ax in fig.axes if ax.get_title().startswith("R = ")][0]
    assert history.get_ylim()[0] >= 0.0  # squared errors are never below 0


def test_regression_steps_lower_R_and_match_torch(static):
    X, y, groups = sine_data()
    net = sine_net()
    ref = copy.deepcopy(net)
    s = explore(net, X, y, groups, eta=0.2, task="regression")
    s.step(25)
    for _ in range(25):
        F.mse_loss(ref(X).reshape(-1), y).backward()
        with torch.no_grad():
            for p in ref.parameters():
                p -= 0.2 * p.grad
                p.grad = None
    assert s.weights() == pytest.approx(naming_order(ref), abs=1e-5)
    assert s.history[-1] < s.history[0]


def test_two_inputs_and_three_inputs(static):
    torch.manual_seed(0)
    net = nn.Sequential(nn.Linear(2, 3), nn.Tanh(), nn.Linear(3, 1))
    X, y = regression_points(net, n=6)
    fig = explore(net, X, y, task="regression").fig
    assert any(ax.get_title().startswith("prediction over the input plane") for ax in fig.axes)
    net3 = make("three_inputs")
    X3, y3 = regression_points(net3, n=5)
    fig3 = explore(net3, X3, y3, task="regression").fig
    assert "3 inputs: no picture of the input plane" in texts(fig3)


def test_binary_one_input_shows_p_against_x(static):
    s = explore(make("one_input"), [-1.0, 0.0, 2.0], [0, 1, 1])
    assert any(ax.get_title() == "p for class 1 against x1" for ax in s.fig.axes)
