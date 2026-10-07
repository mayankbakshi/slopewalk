"""model.py: parsing, the naming order, weights in and out, forward values, the loss and the slopes."""

import copy
import re

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from helpers import NETS, linears, make, naming_order, points

from slopewalk import weight_table
from slopewalk.model import Model, parse_layers


@pytest.mark.parametrize("name", list(NETS))
def test_parse_supported(name):
    net = make(name)
    layers = parse_layers(net)
    assert [L.linear for L in layers] == linears(net)
    assert layers[-1].act is None and layers[-1].out_features == 1
    for L in layers[:-1]:
        assert (L.act is None) == (L.module is None)


@pytest.mark.parametrize(
    "net, fragment",
    [
        (nn.Sequential(nn.Linear(2, 1), nn.Sigmoid()), "must end with nn.Linear(..., 1)"),
        (nn.Sequential(nn.Linear(2, 3), nn.Tanh(), nn.Linear(3, 2)), "must end with nn.Linear(..., 1)"),
        (nn.Sequential(), "must end with nn.Linear(..., 1)"),
        (nn.Sequential(nn.Tanh(), nn.Linear(2, 1)), "unsupported layer Tanh()"),
        (nn.Sequential(nn.Linear(2, 3), nn.Tanh(), nn.Tanh(), nn.Linear(3, 1)), "unsupported layer Tanh()"),
        (nn.Sequential(nn.Linear(2, 3), nn.Dropout(0.1), nn.Linear(3, 1)), "unsupported layer Dropout"),
        (nn.Sequential(nn.Linear(2, 3), nn.BatchNorm1d(3), nn.Linear(3, 1)), "unsupported layer BatchNorm1d"),
        (nn.Sequential(nn.Conv1d(1, 1, 1), nn.Linear(2, 1)), "unsupported layer Conv1d"),
        (nn.Sequential(nn.Linear(2, 3), nn.Softmax(dim=1), nn.Linear(3, 1)), "unsupported layer Softmax"),
    ],
)
def test_parse_rejected_with_a_clear_message(net, fragment):
    with pytest.raises(ValueError, match=re.escape(fragment)) as info:
        parse_layers(net)
    assert "nn.Linear" in str(info.value)  # the message says what is supported


@pytest.mark.parametrize("name", list(NETS))
def test_names_and_order_match_named_parameters(name):
    net = make(name)
    M = Model(net)
    expected = naming_order(net)
    assert M.n_weights == len(expected) == sum(p.numel() for p in net.parameters())
    assert M.names == [f"w{k + 1}" for k in range(len(expected))]
    assert M.get_weights() == expected


def test_weight_table_rows():
    net = make("tanh3")
    rows = weight_table(net)
    assert [r.name for r in rows] == [f"w{k}" for k in range(1, 14)]
    assert (
        rows[0][:4] == ("w1", 1, 1, "x1") and rows[1][:4] == ("w2", 1, 1, "x2") and rows[2][:4] == ("w3", 1, 1, "bias")
    )
    assert rows[3][:4] == ("w4", 1, 2, "x1") and rows[8][:4] == ("w9", 1, 3, "bias")
    assert (
        rows[9][:4] == ("w10", 2, 1, "h1")
        and rows[11][:4] == ("w12", 2, 1, "h3")
        and rows[12][:4] == ("w13", 2, 1, "bias")
    )
    assert rows[0].value == net[0].weight[0, 0].item() and rows[12].value == net[2].bias[0].item()
    rows2 = weight_table(make("two_hidden"))
    assert [r.input for r in rows2 if r.layer == 2][:3] == ["h1,1", "h1,2", "h1,3"]
    assert [r.input for r in rows2 if r.layer == 3] == ["h2,1", "h2,2", "bias"]


@pytest.mark.parametrize("name", list(NETS))
def test_set_and_get_reproduce_the_weights_bit_for_bit(name):
    net = make(name)
    M = Model(net)
    before = [p.detach().clone() for p in net.parameters()]
    w = M.get_weights()
    M.set_weights([v + 1.0 for v in w])
    assert all(not torch.equal(p, b) for p, b in zip(net.parameters(), before))
    M.set_weights(w)
    assert all(torch.equal(p, b) for p, b in zip(net.parameters(), before))
    new = torch.randn(len(w)).tolist()  # float32 values as Python floats
    M.set_weights(new)
    assert M.get_weights() == new


def test_set_weights_keeps_parameters_flags_and_autograd():
    net = make("tanh3")
    M = Model(net)
    net[0].bias.requires_grad_(False)
    params = list(net.parameters())
    M.set_weights([0.1] * 13)
    assert list(net.parameters()) == params  # the same Parameter objects
    assert [p.requires_grad for p in net.parameters()] == [True, False, True, True]
    assert all(p.is_leaf for p in net.parameters())
    X, y = points(net)
    F.binary_cross_entropy_with_logits(net(X).reshape(-1), y).backward()
    assert net[0].weight.grad is not None and net[0].bias.grad is None and net[2].weight.grad is not None


@pytest.mark.parametrize("name", list(NETS))
def test_forward_matches_the_network(name):
    net = make(name)
    M = Model(net)
    X, y = points(net)
    out = M.forward(X, y)
    with torch.no_grad():
        a = net(X).reshape(-1)
        t = X
        for L in M.layers[:-1]:
            z = L.linear(t)
            t = L.module(z) if L.module is not None else z
            z_, h_ = out.hidden[M.layers.index(L)]
            assert torch.allclose(torch.as_tensor(z_), z) and torch.allclose(torch.as_tensor(h_), t)
    assert len(out.hidden) == len(M.hidden)
    assert torch.allclose(torch.as_tensor(out.a), a, atol=1e-6)
    assert torch.allclose(torch.as_tensor(out.p), torch.sigmoid(a), atol=1e-6)
    assert torch.allclose(
        torch.as_tensor(out.loss), F.binary_cross_entropy_with_logits(a, y, reduction="none"), atol=1e-6
    )


@pytest.mark.parametrize("name", list(NETS))
def test_risk_matches_the_torch_loss(name):
    net = make(name)
    M = Model(net)
    X, y = points(net)
    with torch.no_grad():
        expected = F.binary_cross_entropy_with_logits(net(X).reshape(-1), y).item()
    assert M.risk(X, y) == expected
    assert M.risk(X, y) == pytest.approx(M.forward(X, y).loss.mean(), abs=1e-6)  # R is the mean loss over all points
    w = [v * 0.5 for v in M.get_weights()]
    M.risk(X, y, w)
    assert M.get_weights() == pytest.approx(w)  # risk(w) leaves the network at w


@pytest.mark.parametrize("name", list(NETS))
def test_slopes_match_autograd(name):
    net = make(name)
    M = Model(net)
    X, y = points(net)
    g = M.slopes(X, y)
    ref = copy.deepcopy(net)
    F.binary_cross_entropy_with_logits(ref(X).reshape(-1), y).backward()
    expected = naming_order_of_grads(ref)
    assert g.tolist() == pytest.approx(expected, abs=1e-7)


def naming_order_of_grads(net):
    out = []
    for L in linears(net):
        for j in range(L.out_features):
            out += L.weight.grad[j].tolist() + [L.bias.grad[j].item()]
    return out


@pytest.mark.parametrize("name", list(NETS))
def test_slopes_match_central_differences(name):
    net = make(name)
    X, y = points(net)
    g = Model(net).slopes(X, y)
    M64 = Model(copy.deepcopy(net).double())  # finite differences in float64, with the same weights
    X64, y64 = X.double(), y.double()
    w = M64.get_weights()
    h = 1e-5
    fd = []
    for k in range(len(w)):
        up, down = list(w), list(w)
        up[k] += h
        down[k] -= h
        fd.append((M64.risk(X64, y64, up) - M64.risk(X64, y64, down)) / (2 * h))
    assert g.tolist() == pytest.approx(fd, abs=1e-4)


def test_slopes_leave_grads_flags_and_mode_untouched():
    net = make("tanh3")
    X, y = points(net)
    net[0].weight.grad = torch.ones_like(net[0].weight)
    net[2].bias.requires_grad_(False)
    net.eval()
    g = Model(net).slopes(X, y)
    assert torch.equal(net[0].weight.grad, torch.ones_like(net[0].weight))
    assert net[0].bias.grad is None and net[2].weight.grad is None and net[2].bias.grad is None
    assert [p.requires_grad for p in net.parameters()] == [True, True, True, False]
    assert net.training is False
    # a frozen parameter still gets its true slope
    ref = copy.deepcopy(net)
    for p in ref.parameters():
        p.requires_grad_(True)
    assert g.tolist() == pytest.approx(Model(ref).slopes(X, y).tolist(), abs=1e-7)
    net.train()
    Model(net).slopes(X, y)
    assert net.training is True
