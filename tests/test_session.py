"""The session: weight subsets, first-order predictions, the path, the momentum look-ahead and the readouts."""

import copy

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from helpers import make, naming_order, points

from slopewalk import Session, xor_centres
from slopewalk.model import Model

XOR = xor_centres()


@pytest.mark.parametrize(
    "params, names",
    [
        (None, [f"w{k}" for k in range(1, 14)]),
        (["w10", "w11"], ["w10", "w11"]),
        ("w11, w10", ["w10", "w11"]),
        ("w7", ["w7"]),
        ("output", ["w10", "w11", "w12", "w13"]),
        ("layer:1", [f"w{k}" for k in range(1, 10)]),
        ("layer : 2", ["w10", "w11", "w12", "w13"]),
    ],
)
def test_select(params, names):
    M = Model(make("tanh3"))
    assert [M.names[k] for k in M.select(params)] == names


def test_steps_move_only_the_selected_weights():
    net = make("tanh3")
    s = Session(net, *XOR, eta=0.5, params=["w10", "w11"])
    w0 = s.weights()
    g = s.slopes()
    w1 = s.step(1)
    for k in range(13):
        if k in (9, 10):
            assert w1[k] == pytest.approx(w0[k] - 0.5 * g[k], abs=1e-7)
        else:
            assert w1[k] == w0[k]  # frozen weights stay bit for bit
    assert s.weights() == pytest.approx(w1, abs=1e-7)


def test_prediction_is_minus_eta_times_the_squared_slopes():
    s = Session(make("tanh3"), *XOR, eta=0.5)
    g = s.slopes()
    assert s.predict() == pytest.approx(-0.5 * float((g**2).sum()), rel=1e-12)
    s_sub = Session(make("tanh3"), *XOR, eta=0.5, params="output")
    g = s_sub.slopes()
    assert s_sub.predict() == pytest.approx(-0.5 * float((g[9:] ** 2).sum()), rel=1e-12)
    R0 = s.risk()
    s.step(1)
    p = s.predictions[-1]
    assert p.before == pytest.approx(R0) and p.actual == s.history[-1]
    assert p.predicted == pytest.approx(R0 - 0.5 * float((s.slopes(s.w0) ** 2).sum()), abs=1e-9)


def test_prediction_error_shrinks_like_eta_squared():
    s = Session(make("tanh3"), *XOR)
    w0 = s.weights()
    errors = []
    for eta in (0.1, 0.05, 0.025):
        s.reset()
        s.step(1, eta=eta, w=w0)
        p = s.predictions[-1]
        errors.append(abs(p.actual - p.predicted))
    assert errors[1] < errors[0] / 3 and errors[2] < errors[1] / 3  # second order: about a factor of 4


def test_path_and_reset():
    s = Session(make("tanh3"), *XOR, eta=0.5)
    s.step(2)
    assert len(s.path) == 3 and len(s.history) == 3 and len(s.predictions) == 2
    moved = s.weights()
    moved[0] += 0.3
    s.step(1, w=moved)  # a jump made with the sliders starts a new segment of the path
    assert np.isnan(s.path[3]).all() and s.path[4] == pytest.approx(moved)
    s.reset()
    assert s.path == [s.w0] and s.predictions == [] and s.history == [s.risk(s.w0)]


def test_lookahead_matches_a_hand_written_loop_and_leaves_the_network():
    net = make("tanh3")
    X, y, groups = XOR
    s = Session(net, X, y, groups, eta=0.3, beta=0.8)
    before = s.weights()
    look = s.lookahead(steps=7)
    assert s.weights() == before
    for key, beta in (("gd", 0.0), ("momentum", 0.8)):
        ref = copy.deepcopy(net)
        params = list(ref.parameters())
        velocity = [torch.zeros_like(p) for p in params]
        Rs = []
        for _ in range(8):
            R = F.binary_cross_entropy_with_logits(ref(X).reshape(-1), y)
            Rs.append(R.item())
            grads = torch.autograd.grad(R, params)
            with torch.no_grad():
                for p, v, g in zip(params, velocity, grads):
                    v.mul_(beta).sub_(0.3 * g)
                    p.add_(v)
        assert look[key]["R"] == pytest.approx(Rs[: len(look[key]["R"])], abs=1e-5)
        assert look[key]["path"].shape == (8, 13)
    assert look["momentum"]["R"][-1] != look["gd"]["R"][-1]


def test_lookahead_respects_the_subset():
    s = Session(make("tanh3"), *XOR, params=["w10", "w11"])
    path = s.lookahead(steps=5)["momentum"]["path"]
    frozen = [k for k in range(13) if k not in (9, 10)]
    assert np.all(path[:, frozen] == path[0, frozen])


def test_readouts():
    s = Session(make("tanh3"), *XOR, eta=0.5)
    r = s.readouts()
    g = s.slopes()
    assert r["R"] == pytest.approx(s.risk()) and r["grad_norm"] == pytest.approx(float(np.sqrt((g**2).sum())))
    assert r["next_change"] == pytest.approx(-0.5 * float((g**2).sum())) and r["last"] is None
    assert 0 <= r["accuracy"] <= 1
    s.step(1)
    p = s.predictions[-1]
    assert s.readouts()["last"] == (p.predicted - p.before, p.actual - p.before)


def test_case_slopes_one_pass_against_the_offset_oracle():
    from helpers import NETS, case_slope_by_offset

    for name in NETS:
        net = make(name)
        X, y = points(net, n=9)
        M = Model(net)
        flags = [p.requires_grad for p in net.parameters()]
        G = M.case_slopes(X, y)
        assert G.shape == (9, M.n_weights)
        for k in range(M.n_weights):
            assert G[:, k] == pytest.approx(case_slope_by_offset(M, X, y, k), abs=1e-6)
        assert G.mean(0) == pytest.approx(M.slopes(X, y), abs=1e-6)  # the mean is dR/dw
        assert all(p.grad is None for p in net.parameters())
        assert [p.requires_grad for p in net.parameters()] == flags


def test_case_slopes_of_output_weights_follow_the_formula():
    net = make("tanh3")
    X, y, _ = XOR
    M = Model(net)
    G = M.case_slopes(X, y)
    with torch.no_grad():
        h = torch.tanh(net[0](X))
        p = torch.sigmoid(net(X).reshape(-1))
    for j in range(3):
        assert G[:, 9 + j] == pytest.approx(((p - y) * h[:, j]).tolist(), abs=1e-6)  # (p - y) h_j, as on the slide
    assert G[:, 12] == pytest.approx((p - y).tolist(), abs=1e-6)
    Mr = Model(make("tanh3"), task="regression")
    Gr = Mr.case_slopes(X, y)
    with torch.no_grad():
        yhat = Mr.net(X).reshape(-1)
    assert Gr[:, 12] == pytest.approx((2 * (yhat - y)).tolist(), abs=1e-6)


def test_risk_batch_equals_risk_row_by_row():
    for task in ("binary", "regression"):
        net = make("two_hidden")
        X, y = points(net, n=11)
        M = Model(net, task=task)
        w = np.array(M.get_weights())
        W = w[None, :] + 0.3 * np.random.default_rng(0).standard_normal((9, len(w)))
        batch = M.risk_batch(X, y, W)
        assert batch == pytest.approx([M.risk(X, y, list(row)) for row in W], abs=1e-6)
    assert naming_order(net)  # the network is still usable


def test_step_back_undoes_the_last_step():
    s = Session(make("tanh3"), *xor_centres(), eta=0.5)
    w0 = s.weights()
    assert s.step_back() is None  # nothing to undo
    s.step(1)
    w1 = s.weights()
    s.step(2)
    assert len(s.history) == 4 and len(s.predictions) == 3 and len(s.path) == 4
    assert s.step_back() == pytest.approx(s.weights()) and len(s.history) == 3 and len(s.path) == 3
    s.step_back()
    assert s.weights() == pytest.approx(w1) and len(s.history) == 2 and len(s.predictions) == 1 and len(s.path) == 2
    assert s.path[-1] == pytest.approx(w1) and s.history[-1] == pytest.approx(s.risk(w1))
    s.step_back()
    assert s.weights() == pytest.approx(w0) and s.history == [s.risk(w0)] and s.path == [w0] and s.predictions == []


def test_step_back_after_a_slider_move():
    """A slider move after a step is undone with the step; the path keeps its segments consistent."""
    s = Session(make("tanh3"), *xor_centres(), eta=0.5)
    w0 = s.weights()
    s.step(1)
    w1 = s.weights()
    moved = list(w1)
    moved[0] += 0.4
    s.model.set_weights(moved)  # what a slider does
    s.step(1)  # from the moved weights: the path starts a new segment
    assert np.isnan(s.path[2]).all() and s.path[3] == pytest.approx(moved) and len(s.path) == 5
    s.step_back()
    assert s.weights() == pytest.approx(moved) and len(s.path) == 2 and s.path[-1] == pytest.approx(w1)
    s.model.set_weights([v + 0.1 for v in moved])  # another slider move, then back: the step before it is undone
    s.step_back()
    assert s.weights() == pytest.approx(w0) and s.path == [w0] and len(s.history) == 1
