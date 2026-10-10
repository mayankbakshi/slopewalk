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
    plane = [ax for ax in fig.axes if ax.get_title() == "ŷ against x1, and the points"]
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
    assert any(ax.get_title().startswith("ŷ over the input plane") for ax in fig.axes)
    net3 = make("three_inputs")
    X3, y3 = regression_points(net3, n=5)
    fig3 = explore(net3, X3, y3, task="regression").fig  # more inputs than a plane shows: prediction against target
    assert any(ax.get_title().startswith("ŷ against the target y") for ax in fig3.axes)


def test_binary_one_input_shows_p_against_x(static):
    s = explore(make("one_input"), [-1.0, 0.0, 2.0], [0, 1, 1])
    assert any(ax.get_title() == "p for class 1 against x1" for ax in s.fig.axes)


# ---------------------------------------------------------------- several targets

TARGETS = {
    "one_input_two_targets": lambda: nn.Sequential(nn.Linear(1, 3), nn.Tanh(), nn.Linear(3, 2)),
    "three_inputs_two_targets": lambda: nn.Sequential(nn.Linear(3, 4), nn.Tanh(), nn.Linear(4, 2)),
    "linear_three_targets": lambda: nn.Sequential(nn.Linear(2, 3)),
}


def target_points(net, n=12, seed=4):
    """n random points and K targets per point for a network with K outputs."""
    linears = [m for m in net if isinstance(m, nn.Linear)]
    g = torch.Generator().manual_seed(seed)
    return torch.randn(n, linears[0].in_features, generator=g), torch.randn(n, linears[-1].out_features, generator=g)


def wave():
    """16 points on two curves of x: the targets sin(1.5 x) and cos(1.5 x), plus noise, built with torch."""
    g = torch.Generator().manual_seed(0)
    x = torch.linspace(-2, 2, 16)[:, None]
    Y = torch.stack([torch.sin(1.5 * x[:, 0]), torch.cos(1.5 * x[:, 0])], dim=1)
    return x, Y + 0.05 * torch.randn(16, 2, generator=g)


@pytest.mark.parametrize("name", list(TARGETS))
def test_several_targets_numbers(name):
    """R is F.mse_loss over all cases and targets; the slopes match autograd and central differences; each case's
    loss is the mean over its targets, so the per-case slopes average to the slopes, and the backward pass of the
    stages gives the same per-case slopes, with delta a = 2(y-hat - y)/K at the output."""
    from slopewalk.backprop import backward_pass

    torch.manual_seed(0)
    net = TARGETS[name]()
    X, Y = target_points(net)
    M = Model(net, task="regression")
    K = Y.shape[1]
    with torch.no_grad():
        assert M.risk(X, Y) == pytest.approx(F.mse_loss(net(X), Y).item(), abs=1e-6)
        out = M.forward(X, Y)
        assert out.a.shape == (12, K) and out.p is None
        assert out.loss == pytest.approx(((net(X) - Y) ** 2).mean(1).numpy(), abs=1e-6)
    ref = copy.deepcopy(net)
    F.mse_loss(ref(X), Y).backward()
    flat = []
    for m in ref:
        if isinstance(m, nn.Linear):
            for j in range(m.out_features):
                flat += m.weight.grad[j].tolist() + [m.bias.grad[j].item()]
    g = M.slopes(X, Y)
    assert g.tolist() == pytest.approx(flat, abs=1e-6)
    M64 = Model(copy.deepcopy(net).double(), task="regression")
    w = M64.get_weights()
    for k in range(len(w)):
        up, down = list(w), list(w)
        up[k] += 1e-6
        down[k] -= 1e-6
        fd = (M64.risk(X.double(), Y.double(), up) - M64.risk(X.double(), Y.double(), down)) / 2e-6
        assert g[k] == pytest.approx(fd, abs=1e-4)
    cases = M.case_slopes(X, Y)
    assert cases.mean(0) == pytest.approx(g, abs=1e-6)
    bp = backward_pass(M, X, Y)
    assert bp.grads == pytest.approx(cases, abs=1e-6)
    with torch.no_grad():
        assert bp.delta[-1] == pytest.approx((2 * (net(X) - Y) / K).numpy(), abs=1e-6)
    W = torch.tensor(w, dtype=torch.float64)[None, :] + 0.2 * torch.randn(4, len(w), dtype=torch.float64)
    assert M.risk_batch(X, Y, W.numpy()) == pytest.approx([M.risk(X, Y, row.tolist()) for row in W], abs=1e-5)


@pytest.mark.parametrize(
    "y, task, outputs, fragment",
    [
        ("two_columns", None, 2, "y has 2 columns, one per target: pass task='regression'"),
        ("two_columns", "regression", 3, "y has 2 targets per point, but the network has 3 output(s)"),
        ("one_column", "regression", 2, "the network has 2 outputs, so y needs 2 targets per point"),
    ],
)
def test_several_targets_are_checked(y, task, outputs, fragment):
    from slopewalk.data import prepare

    X = torch.randn(5, 2)
    Y = torch.randn(5, 2) if y == "two_columns" else torch.randn(5)
    with pytest.raises(ValueError, match=re.escape(fragment)):
        prepare(X, Y, n_inputs=2, task=task, n_outputs=outputs)
    assert prepare(X, torch.randn(5, 1), n_inputs=2, task="regression").y.shape == (5,)  # one column: one target


def test_several_targets_picture(static):
    """One input, two targets: a prediction box per target, the loss box lists the targets, the panel below shows
    each target's prediction against x1 with its points, the slider groups count the targets from 1."""
    torch.manual_seed(0)
    x, Y = wave()
    s = explore(TARGETS["one_input_two_targets"](), x, Y, task="regression", eta=0.2)
    P = panels(s.fig)
    assert len(P) == 4 and P[0].get_title().startswith("A: mean of 4 points (-1.6):  loss ")
    starts = [t.get_text().split("\n")[0] for t in P[0].texts]
    assert "ŷ1" in starts and "ŷ2" in starts
    loss = [t.get_text() for t in P[0].texts if "\nloss " in t.get_text()][0]
    assert loss.startswith("y1 ") and "\ny2 " in loss and "change\nfrom start\n+0.000" in loss
    curves = [ax for ax in s.fig.axes if ax.get_title() == "ŷ of each target against x1,\nand the points"]
    assert len(curves) == 1 and [ln.get_label() for ln in curves[0].lines] == ["ŷ1", "ŷ2"]
    assert [s.model.unit_title(1, j) for j in range(2)] == ["output unit 1", "output unit 2"]
    R0 = s.risk()
    s.step(50)
    assert s.risk() < R0


@pytest.mark.parametrize("inputs, outputs", [(3, 1), (2, 2), (3, 2)])
def test_more_inputs_or_targets_show_prediction_against_target(inputs, outputs, static):
    """Three or more inputs, or two inputs with several targets: ŷ against the target y, with
    the diagonal."""
    torch.manual_seed(0)
    X = torch.rand(30, inputs) * 4 - 2
    Y = torch.randn(30, outputs) if outputs > 1 else X[:, 0] * X[:, 1]
    net = nn.Sequential(nn.Linear(inputs, 3), nn.Tanh(), nn.Linear(3, outputs))
    s = explore(net, X, Y, task="regression", views=("hidden", "cases", "slice", "line", "momentum"))
    ax = [a for a in s.fig.axes if a.get_title().startswith("ŷ against the target y")]
    assert len(ax) == 1
    marks = [c for c in ax[0].collections if len(c.get_offsets()) == 30]
    assert len(marks) == outputs


def test_several_targets_backward_pass(static):
    torch.manual_seed(0)
    x, Y = wave()
    s = explore(TARGETS["one_input_two_targets"](), x, Y, task="regression", views="backprop", backprop_step=1)
    assert s.backprop_stages[1].text == "at the output: δak = ∂ℓ/∂ak = 2(ŷk - yk)/K (ℓ is the mean over the K targets)"
    from slopewalk.backprop import derivation_lines
    from slopewalk.picture import make_frame

    lines = [t for _, t in derivation_lines(s, make_frame(s), 0)]
    assert lines[0].startswith("forward: ŷ = (") and lines[1].startswith("targets: y = (")
    assert lines[2].startswith("δa1 = 2(ŷ1 - y1)/2 = 2(") and lines[3].startswith("δa2 = 2(ŷ2 - y2)/2 = 2(")
    assert any(t.startswith("w7: δa1 × h1 = ") for t in lines)  # w1 to w6: the first layer
    deltas = [t.get_text().split("\n")[0] for t in panels(s.fig)[0].texts if t.get_text().startswith("δa")]
    assert deltas == ["δa1", "δa2"]


def test_targets_are_readable(static):
    """Each target's line and marker share one legend entry; the second target's markers are hollow; the hint says
    that targets count from 1; one group gets no letter in the prediction-against-target panel, several get a cross
    and a letter beside it."""
    from matplotlib.colors import to_rgba

    torch.manual_seed(0)
    x, Y = wave()
    s = explore(TARGETS["one_input_two_targets"](), x, Y, task="regression")
    ax = [a for a in s.fig.axes if a.get_title() == "ŷ of each target against x1,\nand the points"][0]
    assert [t.get_text() for t in ax.get_legend().get_texts()] == ["ŷ1 and y1", "ŷ2 and y2"]
    first, second = [c for c in ax.collections if len(c.get_offsets()) == 16]
    assert tuple(second.get_facecolor()[0]) == to_rgba(s.theme.paper) and first.get_sizes()[0] >= 36
    X = torch.rand(30, 3) * 4 - 2
    one = explore(nn.Sequential(nn.Linear(3, 2), nn.Tanh(), nn.Linear(2, 1)), X, X[:, 0], task="regression")
    parity = [a for a in one.fig.axes if a.get_title().startswith("ŷ against the target y")][0]
    assert not parity.texts  # one group: no letter
    two = explore(
        nn.Sequential(nn.Linear(3, 2), nn.Tanh(), nn.Linear(2, 1)), X, X[:, 0], [0] * 15 + [1] * 15, task="regression"
    )
    parity = [a for a in two.fig.axes if a.get_title().startswith("ŷ against the target y")][0]
    assert [t.get_text() for t in parity.texts] == ["A", "B"]
