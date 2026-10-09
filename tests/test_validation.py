"""Checks on the inputs of explore: each mistake gives a clear ValueError, and the accepted forms work."""

import math
import re

import numpy as np
import pytest
import torch
import torch.nn as nn
from helpers import make

from slopewalk import Session, explore, xor_centres
from slopewalk.data import prepare


@pytest.fixture(autouse=True)
def static(monkeypatch):
    import matplotlib.pyplot as plt

    monkeypatch.setenv("NB_STATIC", "1")
    monkeypatch.setattr(plt, "show", lambda *a, **k: None)
    yield
    plt.close("all")


XOR = xor_centres()


@pytest.mark.parametrize(
    "X, y, groups, fragment",
    [
        ([["a", "b"], ["c", "d"]], [0, 1], None, "X must hold numbers"),
        ([[0, 0], [1]], [0, 1], None, "X must hold numbers"),
        (np.zeros((2, 2, 2)), [0, 1], None, "n x d array of points"),
        (np.zeros((0, 2)), [], None, "X has no points"),
        (np.zeros((3, 3)), [0, 1, 0], None, "have 3 inputs, but the network has 2"),
        ([1.0, 2.0, 3.0], [0], None, "X has 3 numbers, but the network has 2 inputs"),
        ([[0, 0], [math.nan, 1]], [0, 1], None, "X contains NaN or inf at 1 point(s), the first at index 1"),
        ([[0, 0], [1, math.inf]], [0, 1], None, "X contains NaN or inf"),
        ([[0, 0], [1, 1]], [0, 1, 1], None, "y has 3 values for 2 points"),
        ([[0, 0], [1, 1]], [0, math.nan], None, "y contains NaN or inf"),
        ([[0, 0], [1, 1]], [0, 2], None, "pass task='regression'"),
        ([[0, 0], [1, 1]], [0, 0.5], None, "y has values other than 0 and 1 (for example 0.5)"),
        ([[0, 0], [1, 1]], [0, 1], [0], "groups has 1 values for 2 points"),
        ([[0, 0], [1, 1]], [0, 1], [0, 0.5], "groups must hold one whole number per point"),
    ],
)
def test_bad_points(X, y, groups, fragment):
    with pytest.raises(ValueError, match=re.escape(fragment)):
        explore(make("tanh3"), X, y, groups)


def test_binary_task_with_other_labels():
    with pytest.raises(ValueError, match="other than 0 and 1"):
        explore(make("tanh3"), [[0, 0], [1, 1]], [0, 2], task="binary")


@pytest.mark.parametrize(
    "kwargs, fragment",
    [
        (dict(task="classify"), "task must be 'binary', 'multiclass' or 'regression'"),
        (dict(eta=0), "eta must be a positive number"),
        (dict(eta=math.nan), "eta must be a positive number"),
        (dict(beta=1.0), "beta"),
        (dict(lookahead=0), "lookahead must be at least 1"),
        (dict(views=("slice", "surface")), "unknown view(s) ['surface']"),
        (dict(case_slopes="w99"), "'w99' is not a weight of this network"),
        (dict(slice_weights=("w1", "w1")), "one weight or two different weights"),
        (dict(slice_weights=("w1", "w2", "w3")), "one weight or two different weights"),
        (dict(params=["w14"]), "does not name weights of this network"),
        (dict(params="layer:3"), "this network has layers 1 to 2"),
        (dict(params=[]), "does not name weights"),
    ],
)
def test_bad_options(kwargs, fragment):
    with pytest.raises(ValueError, match=re.escape(fragment)):
        explore(make("tanh3"), *XOR, **kwargs)


def test_bad_networks():
    X, y, groups = XOR
    with pytest.raises(ValueError, match="has no bias"):
        explore(nn.Sequential(nn.Linear(2, 3, bias=False), nn.Tanh(), nn.Linear(3, 1)), X, y, groups)
    with pytest.raises(ValueError, match=r"float16.*net\.float\(\)"):
        explore(make("tanh3").half(), X, y, groups)
    mixed = make("tanh3")
    mixed[0].double()
    with pytest.raises(ValueError, match="float32, float64"):
        explore(mixed, X, y, groups)
    meta = make("tanh3")
    try:
        meta = meta.to("meta")
    except (RuntimeError, NotImplementedError, TypeError):
        pytest.skip("this torch has no meta device")
    with pytest.raises(ValueError, match=r"on the CPU.*net\.cpu\(\)"):
        explore(meta, X, y, groups)


@pytest.mark.parametrize(
    "X, y",
    [
        ([(-1, 1), (1, 1)], [1, 0]),  # lists of tuples and ints
        (np.array([[-1, 1], [1, 1]], dtype=np.int64), np.array([True, False])),  # integer X, boolean y
        (
            torch.tensor([[-1.0, 1.0], [1.0, 1.0]], requires_grad=True),
            torch.tensor([[1.0], [0.0]]),
        ),  # y of shape (n, 1)
        (np.array([[-1, 1], [1, 1]], dtype=np.float64), [1.0, 0.0]),
    ],
)
def test_accepted_forms(X, y):
    s = explore(make("logistic"), X, y)
    assert s.data.X.dtype == torch.float32 and s.data.y.tolist() == [1.0, 0.0]
    assert s.data.X.tolist() == [[-1.0, 1.0], [1.0, 1.0]]


def test_flat_X_for_one_input_is_many_points():
    s = explore(make("one_input"), [-1.0, 0.0, 2.0], [0, 1, 1])
    assert s.data.X.shape == (3, 1) and s.data.G == 3
    one = prepare([0.5, 0.5], [1], n_inputs=2)
    assert one.X.shape == (1, 2)


def test_float64_network_everywhere():
    net = make("tanh3").double()
    X, y, groups = XOR
    s = explore(net, X, y, groups)
    assert s.data.X.dtype == torch.float64 and s.model.dtype == torch.float64
    with torch.no_grad():
        R = nn.functional.binary_cross_entropy_with_logits(net(X.double()).reshape(-1), y.double()).item()
    assert s.risk() == pytest.approx(R, abs=1e-12)
    s.step(3)
    assert s.fig is not None and len(s.history) == 4
    assert s.draw() is s.fig


def test_default_groups():
    assert prepare(np.zeros((10, 2)), [0, 1] * 5).groups.tolist() == [0, 1] * 5
    reg = prepare(np.arange(12.0)[::-1, None], np.arange(12.0), n_inputs=1, task="regression")
    assert reg.groups.tolist() == [3, 3, 3, 2, 2, 2, 1, 1, 1, 0, 0, 0]  # four groups of neighbours along x1
    two = prepare(np.zeros((12, 2)), np.arange(12.0), task="regression")
    assert two.groups.tolist() == [0] * 12


def test_session_without_drawing():
    s = Session(make("tanh3"), *XOR)
    assert s.task == "binary" and s.fig is None and s.controller is None
