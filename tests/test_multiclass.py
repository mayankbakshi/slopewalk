"""Multiclass: a network with K outputs, the scores of the classes 0 to K-1; R is the cross-entropy."""

import copy

import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from helpers import MULTICLASS, case_slope_by_offset, class_points, make

from slopewalk import clusters
from slopewalk.data import prepare
from slopewalk.model import Model


@pytest.mark.parametrize("name", list(MULTICLASS))
def test_risk_is_the_cross_entropy(name):
    net = make(name)
    X, y = class_points(net)
    M = Model(net)
    assert M.task == "multiclass" and M.n_outputs == net[-1].out_features
    expected = F.cross_entropy(net(X), y.long()).item()
    assert M.risk(X, y) == pytest.approx(expected, abs=1e-6)


@pytest.mark.parametrize("name", list(MULTICLASS))
def test_slopes_match_autograd_and_finite_differences(name):
    net = make(name)
    X, y = class_points(net)
    M = Model(net)
    g = M.slopes(X, y)
    ref = copy.deepcopy(net)
    F.cross_entropy(ref(X), y.long()).backward()
    auto = []  # in the order of the names
    for m in ref:
        if isinstance(m, nn.Linear):
            for j in range(m.out_features):
                auto += m.weight.grad[j].tolist() + [m.bias.grad[j].item()]
    assert g.tolist() == pytest.approx(auto, abs=1e-6)
    M64 = Model(copy.deepcopy(net).double())
    X64, y64 = X.double(), y.double()
    w = M64.get_weights()
    h = 1e-6
    for k in range(len(w)):
        up, down = list(w), list(w)
        up[k] += h
        down[k] -= h
        fd = (M64.risk(X64, y64, up) - M64.risk(X64, y64, down)) / (2 * h)
        assert g[k] == pytest.approx(fd, abs=1e-4)


@pytest.mark.parametrize("name", list(MULTICLASS))
def test_case_slopes_average_to_the_slopes(name):
    net = make(name)
    X, y = class_points(net)
    M = Model(net)
    cases = M.case_slopes(X, y)
    assert cases.shape == (X.shape[0], M.n_weights)
    assert cases.mean(0) == pytest.approx(M.slopes(X, y), abs=1e-6)
    for k in (0, M.n_weights - 1):
        assert cases[:, k] == pytest.approx(case_slope_by_offset(M, X, y, k), abs=1e-6)


def test_risk_batch_matches_risk():
    net = make("three_classes")
    X, y = class_points(net, n=11)
    M = Model(net)
    w = np.asarray(M.get_weights())
    W = w[None, :] + 0.3 * np.random.default_rng(0).standard_normal((5, len(w)))
    batch = M.risk_batch(X, y, W)
    assert batch == pytest.approx([M.risk(X, y, list(row)) for row in W], abs=1e-6)


def test_forward_values_and_accuracy():
    net = make("three_classes")
    X, y = class_points(net, n=12)
    M = Model(net)
    out = M.forward(X, y)
    a = net(X).detach()
    assert out.a.shape == (12, 3) and out.p.shape == (12, 3)
    assert out.a == pytest.approx(a.numpy(), abs=1e-6)
    assert out.p.sum(1) == pytest.approx(np.ones(12), abs=1e-6)
    assert out.loss == pytest.approx(F.cross_entropy(a, y.long(), reduction="none").numpy(), abs=1e-6)
    assert M.accuracy(X, y) == pytest.approx(float((a.argmax(1) == y.long()).float().mean()))


def test_names_and_titles():
    M = Model(make("three_classes"))
    assert M.n_weights == 9 + 12 and M.names[-1] == "w21"
    assert [M.unit_title(1, j) for j in range(3)] == [f"output unit, class {j}" for j in range(3)]
    assert Model(make("tanh3")).unit_title(1, 0) == "output unit"


def test_task_from_the_outputs():
    net = make("three_classes")
    X, y = class_points(net)
    assert prepare(X, y, n_inputs=2, n_outputs=3).task == "multiclass"
    D = prepare(X, y, n_inputs=2, n_outputs=3)
    assert D.n_classes == 3
    two = make("two_outputs")
    assert prepare(X, (y > 0).float(), n_inputs=2, n_outputs=2).task == "multiclass"
    assert Model(two).task == "multiclass"
    assert prepare(X, (y > 0).float(), n_inputs=2).n_classes == 2


def test_default_groups_are_the_classes():
    X, y, _ = clusters([(-1, 1), (1, 1), (0, -1)], [0, 1, 2], noise=0.3, n_per=10)
    D = prepare(X, y, n_inputs=2, n_outputs=3)
    assert D.G == 3 and [int(D.y[sel].unique().item()) for sel in D.gsel] == [0, 1, 2]


@pytest.mark.parametrize(
    "y, n_outputs, task, fragment",
    [
        (
            [0, 1, 3],
            3,
            None,
            "label 3, but the network has 3 outputs, one per class: the labels must be whole numbers from 0 to 2",
        ),
        ([0, 1, 1.5], 3, None, "label 1.5"),
        ([0, -1, 1], 3, None, "label -1"),
        ([0, 1, 2], 3, "binary", "has 3 outputs, one score per class, which is task='multiclass'"),
        ([0, 1, 2], 3, "regression", "For task='regression', end it with nn.Linear(..., 1)"),
        ([0, 1, 1], 1, "multiclass", "needs one output per class"),
        ([0, 1, 2], 1, None, "For 3 classes, end the network with nn.Linear(..., 3) and use the labels 0 to 2"),
    ],
)
def test_labels_and_task_are_checked(y, n_outputs, task, fragment):
    X = [[0, 0], [1, 1], [2, 2]]
    with pytest.raises(ValueError, match="slopewalk: ") as e:
        prepare(X, y, n_inputs=2, task=task, n_outputs=n_outputs)
    assert fragment.lower() in str(e.value).lower()


def test_more_than_six_classes_are_refused():
    with pytest.raises(ValueError, match="up to 6 classes"):
        Model(nn.Sequential(nn.Linear(2, 7)))


def test_float64_network():
    net = make("three_classes").double()
    X, y = class_points(net)
    M = Model(net)
    assert M.risk(X.double(), y.double()) == pytest.approx(F.cross_entropy(net(X.double()), y.long()).item(), abs=1e-12)


def test_one_hot_targets_are_not_needed():
    """The labels stay a float tensor in the session, like binary labels; the loss converts them to class indices."""
    net = make("three_classes")
    X, y = class_points(net)
    D = prepare(X, y, n_inputs=2, n_outputs=3)
    assert D.y.dtype == torch.float32 and set(D.y.tolist()) <= {0.0, 1.0, 2.0}
