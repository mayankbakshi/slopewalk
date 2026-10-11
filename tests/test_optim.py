"""The rule of a step: gradient descent, momentum and Adam, with weight decay, checked against torch.optim; the
first-order prediction as slopes times step; back one step and reset restore the rule's state; test data."""

import copy
import re

import matplotlib.pyplot as plt
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from helpers import make

from slopewalk import Session, explore, xor_centres
from slopewalk.optim import Optimizer


@pytest.fixture
def static(monkeypatch):
    monkeypatch.setenv("NB_STATIC", "1")
    monkeypatch.setattr(plt, "show", lambda *a, **k: None)
    yield
    plt.close("all")


def torch_run(net, X, y, make_opt, steps):
    """The weights after `steps` full-batch steps of a torch optimizer on a copy of net, in the order of the names."""
    ref = copy.deepcopy(net)
    opt = make_opt(ref.parameters())
    for _ in range(steps):
        opt.zero_grad()
        F.binary_cross_entropy_with_logits(ref(X).reshape(-1), y).backward()
        opt.step()
    out = []
    for m in ref:
        if isinstance(m, nn.Linear):
            for j in range(m.out_features):
                out += m.weight[j].tolist() + [m.bias[j].item()]
    return out


@pytest.mark.parametrize(
    "kind, decay, make_opt",
    [
        ("gd", 0.0, lambda p: torch.optim.SGD(p, lr=0.3)),
        ("gd", 0.05, lambda p: torch.optim.SGD(p, lr=0.3, weight_decay=0.05)),
        ("momentum", 0.0, lambda p: torch.optim.SGD(p, lr=0.3, momentum=0.9)),
        ("momentum", 0.02, lambda p: torch.optim.SGD(p, lr=0.3, momentum=0.9, weight_decay=0.02)),
        ("adam", 0.0, lambda p: torch.optim.Adam(p, lr=0.3, betas=(0.9, 0.999))),
        ("adam", 0.02, lambda p: torch.optim.Adam(p, lr=0.3, betas=(0.9, 0.999), weight_decay=0.02)),
    ],
)
def test_steps_match_torch_optim(kind, decay, make_opt):
    X, y, groups = xor_centres()
    net = make("tanh3")
    expected = torch_run(net, X, y, make_opt, 6)
    s = Session(net, X, y, groups, eta=0.3, optimizer=kind, weight_decay=decay)
    s.step(6)
    assert s.weights() == pytest.approx(expected, abs=2e-5)
    assert len(s.history) == 7 and all(p.before == h for p, h in zip(s.predictions, s.history))


def test_prediction_is_slopes_times_step():
    """The first-order prediction of a step is g . dw: for plain gradient descent -eta |g|^2, as before."""
    X, y, groups = xor_centres()
    plain = Session(make("tanh3"), X, y, groups, eta=0.5)
    g = plain.slopes()
    assert plain.predict() == pytest.approx(-0.5 * float((g**2).sum()))
    adam = Session(make("tanh3"), X, y, groups, eta=0.1, optimizer="adam")
    g = adam.slopes()
    dw = adam.next_step()
    assert adam.predict() == pytest.approx(sum(g[k] * dw[k] for k in dw))
    assert sorted(dw) == list(range(13))
    assert all(abs(abs(d) - 0.1) < 1e-4 for k, d in dw.items() if abs(g[k]) > 1e-4)  # a first Adam step: eta each
    w = adam.weights()
    adam.step(1)
    assert adam.weights() == pytest.approx([w[k] + dw[k] for k in range(13)], abs=1e-6)  # float32 weights
    assert adam.predictions[-1].predicted - adam.predictions[-1].before == pytest.approx(sum(g[k] * dw[k] for k in dw))


def test_back_one_step_and_reset_restore_the_state():
    X, y, groups = xor_centres()
    fresh = Session(make("tanh3"), X, y, groups, eta=0.1, optimizer="adam")
    fresh.step(3)
    s = Session(make("tanh3"), X, y, groups, eta=0.1, optimizer="adam")
    s.step(4)
    s.step_back()
    s.step_back()
    s.step(1)
    assert s.weights() == pytest.approx(fresh.weights(), abs=1e-6) and s.optimizer.t == 3  # float32 weights
    s.reset()
    assert s.optimizer.t == 0 and s.optimizer.m == {} and s.history_test == [None]
    s.step(3)
    assert s.weights() == pytest.approx(fresh.weights(), abs=1e-6)


def test_only_the_weights_that_move_move():
    X, y, groups = xor_centres()
    s = Session(make("tanh3"), X, y, groups, eta=0.2, optimizer="adam", params="output")
    w0 = s.weights()
    s.step(3)
    w = s.weights()
    assert w[:9] == w0[:9] and all(w[k] != w0[k] for k in range(9, 13))
    assert set(s.optimizer.m) == set(range(9, 13))


def test_the_line_view_follows_the_step(static):
    """R along the next step: the point at t = eta is R after the step the optimizer takes."""
    X, y, groups = xor_centres()
    s = explore(make("tanh3"), X, y, groups, eta=0.2, optimizer="momentum", views="line")
    s.step(2)
    ax = [a for a in s.draw().axes if a.get_title().startswith("R along the next step")][0]
    label = [ln.get_label() for ln in ax.lines if ln.get_label().startswith("step of eta: R ")][0]
    s.step(1)
    assert label == f"step of eta: R {s.history[-1]:.4f}" and "momentum step" in ax.get_xlabel()


def test_checks():
    X, y, groups = xor_centres()
    with pytest.raises(ValueError, match="optimizer must be 'gd', 'momentum' or 'adam'"):
        Session(make("tanh3"), X, y, groups, optimizer="sgd")
    with pytest.raises(ValueError, match="weight_decay must be a number from 0 up"):
        Session(make("tanh3"), X, y, groups, weight_decay=-0.1)
    with pytest.raises(ValueError, match="beta2"):
        Session(make("tanh3"), X, y, groups, beta2=1.0)
    opt = Optimizer("momentum", beta=0.5)
    dw = opt.step([1.0, -2.0], [0.0, 0.0], [0, 1], eta=0.1)
    assert dw == pytest.approx({0: -0.1, 1: 0.2})
    assert opt.step([1.0, -2.0], [0.0, 0.0], [0, 1], eta=0.1) == pytest.approx({0: -0.15, 1: 0.3})
    assert opt.direction([0.0, 0.0], [0.0, 0.0], [0, 1], eta=0.1) == pytest.approx({0: -0.075, 1: 0.15})  # no change
    assert opt.velocity == pytest.approx({0: -0.15, 1: 0.3})


def test_controls_for_the_optimizer(monkeypatch):
    ipywidgets = pytest.importorskip("ipywidgets")
    import IPython.display

    monkeypatch.delenv("NB_STATIC", raising=False)
    monkeypatch.setattr(IPython.display, "display", lambda *a, **k: None)
    X, y, groups = xor_centres()
    plain = explore(make("tanh3"), X, y, groups).controller
    assert not hasattr(plain, "opt_pick")  # nothing asked for: no row
    s = explore(make("tanh3"), X, y, groups, eta=0.1, optimizer="adam", views="momentum")
    C = s.controller
    assert C.opt_pick.value == "adam" and C.decay.value == 0.0 and isinstance(C.opt_beta, ipywidgets.FloatSlider)
    C.decay.value = 0.01
    assert s.optimizer.weight_decay == 0.01 and "weight decay 0.01" in C.message.value
    C.opt_pick.value = "momentum"
    assert s.optimizer.kind == "momentum" and s.optimizer.t == 0 and "Steps by momentum" in C.message.value
    C.opt_beta.value = 0.5
    assert s.beta == 0.5 and C.beta.value == 0.5  # the momentum view shares beta
    C.beta.value = 0.7
    assert s.optimizer.beta == 0.7 and C.opt_beta.value == 0.7
    C.buttons["gradient step"].click()
    assert len(s.history) == 2 and s.optimizer.velocity
    decay_only = explore(make("tanh3"), X, y, groups, weight_decay=0.05).controller
    assert decay_only.opt_pick.value == "gd" and decay_only.decay.value == 0.05


# ---------------------------------------------------------------- test data


def test_test_data(static):
    X, y, groups = xor_centres()
    Xt, yt, _ = xor_centres(noise=0.3, n_per=10, seed=5)
    s = explore(make("tanh3"), X, y, groups, eta=0.5, X_test=Xt, y_test=yt)
    assert s.test.n == 40 and s.history_test == [s.risk_test(s.w0)]
    r = s.readouts()
    assert r["R_test"] == pytest.approx(F.binary_cross_entropy_with_logits(make("tanh3")(Xt).reshape(-1), yt).item())
    assert 0 <= r["accuracy_test"] <= 1
    s.step(5)
    assert len(s.history_test) == 6 and s.history_test[-1] == pytest.approx(s.risk_test())
    fig = s.draw()
    history = [ax for ax in fig.axes if ax.get_title().startswith("R = ")][0]
    assert any(ln.get_label() == "R on the test data" for ln in history.lines)
    plane = [ax for ax in fig.axes if ax.get_title().startswith("p for class 1 over the input plane")][0]
    crosses = [c for c in plane.collections if len(c.get_offsets()) == 40]
    assert len(crosses) == 1 and "test data" in [t.get_text() for t in plane.get_legend().get_texts()]
    s.step_back()
    assert len(s.history_test) == 5
    s.reset()
    assert s.history_test == [s.risk_test(s.w0)]


def test_test_data_is_checked():
    X, y, groups = xor_centres()
    with pytest.raises(ValueError, match="give X_test and y_test together"):
        Session(make("tanh3"), X, y, groups, X_test=X)
    with pytest.raises(ValueError, match=re.escape("y has 3 values for 4 points")):
        Session(make("tanh3"), X, y, groups, X_test=X, y_test=[0, 1, 1])
    with pytest.raises(ValueError, match="other than 0 and 1"):
        Session(make("tanh3"), X, y, groups, X_test=X, y_test=[0, 1, 2, 1])


def test_test_data_for_regression_and_classes(static):
    torch.manual_seed(0)
    x = torch.linspace(-2, 2, 16)[:, None]
    yr = torch.sin(1.5 * x[:, 0])
    s = explore(
        nn.Sequential(nn.Linear(1, 3), nn.Tanh(), nn.Linear(3, 1)),
        x,
        yr,
        task="regression",
        X_test=x[::2],
        y_test=yr[::2],
    )
    r = s.readouts()
    assert r["R_test"] is not None and r["accuracy_test"] is None
    ax = [a for a in s.fig.axes if a.get_title() == "ŷ against x1, and the points"][0]
    assert any(c.get_label() == "test data" for c in ax.collections)
    X3 = torch.rand(30, 3) * 4 - 2
    y3 = X3[:, 0] * X3[:, 1]
    s3 = explore(
        nn.Sequential(nn.Linear(3, 3), nn.Tanh(), nn.Linear(3, 1)),
        X3,
        y3,
        task="regression",
        X_test=X3[:5],
        y_test=y3[:5],
    )
    parity = [a for a in s3.fig.axes if a.get_title().startswith("ŷ against the target y")][0]
    assert any(c.get_label() == "test data" for c in parity.collections)
    from slopewalk import clusters

    Xm, ym, gm = clusters([(-1, 1), (1, 1), (0, -1)], [0, 1, 2])
    Xt, yt, _ = clusters([(-1, 1), (1, 1), (0, -1)], [0, 1, 2], noise=0.2, n_per=4)
    sm = explore(nn.Sequential(nn.Linear(2, 3), nn.Tanh(), nn.Linear(3, 3)), Xm, ym, gm, X_test=Xt, y_test=yt)
    plane = [a for a in sm.fig.axes if a.get_title().startswith("the most likely class")][0]
    assert any(c.get_label() == "test data" for c in plane.collections)
    assert sm.readouts()["accuracy_test"] is not None
