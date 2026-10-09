"""The backward pass step by step: its numbers against autograd, its stages, its drawing, widgets and recording."""

import re

import matplotlib.pyplot as plt
import numpy as np
import pytest
import torch
import torch.nn.functional as F
from helpers import NETS, make, points, regression_points
from render_examples import reference_net

from slopewalk import Session, explore, xor_centres
from slopewalk.backprop import backprop_stages, backward_pass, derivation_lines, traced_case
from slopewalk.draw import THEME, width_share
from slopewalk.model import Model

XOR = xor_centres()


@pytest.fixture
def static(monkeypatch):
    monkeypatch.setenv("NB_STATIC", "1")
    monkeypatch.setattr(plt, "show", lambda *a, **k: None)
    yield
    plt.close("all")


def autograd_values(model, X, y):
    """dl_i/dz and dl_i/dh for every layer and case, from autograd: the oracle for the backward pass."""
    zs, hs = [], []
    h = X
    for L in model.layers:
        z = F.linear(h, L.linear.weight.detach(), L.linear.bias.detach())
        if not z.requires_grad:
            z.requires_grad_(True)
        h = L.module(z) if L.module is not None else z
        zs.append(z)
        hs.append(h)
    losses = model.loss(h.reshape(-1), y, reduction="none")
    grads = torch.autograd.grad(losses.sum(), zs + hs[:-1])
    return [g.numpy() for g in grads[: len(zs)]], [g.numpy() for g in grads[len(zs) :]]


@pytest.mark.parametrize("task", ["binary", "regression"])
@pytest.mark.parametrize("name", [n for n in NETS if n != "wide"])
def test_backward_pass_matches_autograd(name, task):
    net = make(name)
    X, y = points(net, n=9) if task == "binary" else regression_points(net)
    M = Model(net, task=task)
    bp = backward_pass(M, X, y)
    dz, dh = autograd_values(M, X, y)
    for l in range(len(M.layers)):
        assert bp.delta[l] == pytest.approx(dz[l], abs=1e-6)
    for l in range(len(M.layers) - 1):
        assert bp.dh[l] == pytest.approx(dh[l], abs=1e-6)
    assert bp.grads == pytest.approx(M.case_slopes(X, y), abs=1e-6)
    assert bp.grads.mean(0) == pytest.approx(M.slopes(X, y), abs=1e-6)  # the average is dR/dw


def test_relu_derivative_at_zero_matches_autograd():
    net = make("relu4")
    with torch.no_grad():
        net[0].bias[0] = 0.0
        net[0].weight[0] = 0.0  # z1 = 0 for every case
    X, y = points(net)
    M = Model(net)
    bp = backward_pass(M, X, y)
    dz, _ = autograd_values(M, X, y)
    assert bp.delta[0] == pytest.approx(dz[0], abs=1e-7) and np.all(bp.deriv[0][:, 0] == 0)


def test_stages():
    kinds = [(s.kind, s.layer) for s in backprop_stages(Model(make("tanh3")))]
    assert kinds == [
        ("forward", None),
        ("output", 1),
        ("weights", 1),
        ("to_h", 0),
        ("delta", 0),
        ("weights", 0),
        ("average", None),
    ]
    assert len(backprop_stages(Model(make("logistic")))) == 4
    assert len(backprop_stages(Model(make("two_hidden")))) == 10
    texts = [s.text for s in backprop_stages(Model(make("tanh3")))]
    assert "p - y" in texts[1] and "through tanh" in texts[4]
    assert "2(ŷ - y)" in backprop_stages(Model(make("tanh3"), task="regression"))[1].text


def test_derivation_multiplies_out():
    s = Session(reference_net(40), *XOR, views=("backprop",))
    M = s.model
    bp = backward_pass(M, s.data.X, s.data.y)
    lines = dict(derivation_lines(s, type("F", (), {"bp": bp})(), 0))
    text = "\n".join(t for _, t in derivation_lines(s, type("F", (), {"bp": bp})(), 0))
    a = bp.z[-1][0, 0]
    p = 1 / (1 + np.exp(-a))
    assert f"δa = p - y = {p:.3f} - 1 = {p - 1:+.3f}" in text
    assert f"w10: δa × h1 = {p - 1:+.3f} × {bp.inputs[1][0, 0]:+.3f} = {bp.grads[0, 9]:+.3f}" in text
    assert f"= {bp.delta[0][0, 0]:+.3f}" in text and "tanh′(z1)" in text
    assert lines  # every stage has at least one line
    stages_with_lines = {st for st, _ in derivation_lines(s, type("F", (), {"bp": bp})(), 0)}
    assert stages_with_lines == set(range(len(s.backprop_stages)))


def test_traced_case_is_nearest_the_group_mean():
    s = Session(make("tanh3"), *xor_centres(noise=0.35, n_per=20))
    i = traced_case(s, 1)
    X = s.data.X.numpy()
    idx = np.nonzero(s.data.gsel[1])[0]
    d = ((X[idx] - X[idx].mean(0)) ** 2).sum(1)
    assert i == idx[np.argmin(d)]


def backward_boxes(ax):
    return [t for t in ax.texts if t.get_fontsize() == THEME.backprop_size and t.get_color() == THEME.backprop_colour]


def network_panels(fig):
    return [ax for ax in fig.axes if re.match(r"^[A-Z]: ", ax.get_title())]


def test_pictures_of_every_stage(static):
    s = explore(reference_net(40), *XOR, views="backprop")
    expected_boxes = [0, 1, 1, 4, 7, 7, 7]  # delta_a, then dl/dh1..3, then delta1..3
    for k, n_boxes in enumerate(expected_boxes):
        s.backprop_step = k
        fig = s.draw()
        panels = network_panels(fig)
        assert len(panels) == 4
        assert [len(backward_boxes(ax)) for ax in panels] == [n_boxes] * 4
        labels = [t for t in panels[0].texts if t.get_fontsize() == THEME.edge_label_size]
        kind = s.backprop_stages[k].kind
        assert (
            len(labels)
            == {"forward": 13, "output": 0, "weights": 4 if k == 2 else 9, "to_h": 3, "delta": 0, "average": 13}[kind]
        )
        derivation = [ax for ax in fig.axes if ax.get_title(loc="left").startswith("backward pass of A")]
        assert len(derivation) == 1 and f"stage {k} of 6" in derivation[0].get_title(loc="left")
        plt.close("all")


def test_slope_labels_are_the_group_means(static):
    s = explore(reference_net(40), *XOR, views="backprop", backprop_step=2)
    ax = network_panels(s.fig)[0]
    labels = sorted(t.get_text() for t in ax.texts if t.get_fontsize() == THEME.edge_label_size)
    bp = backward_pass(s.model, s.data.X, s.data.y)
    assert labels == sorted(f"{bp.grads[0, k]:+.3f}" for k in (9, 10, 11, 12))


def test_options(static):
    s = explore(reference_net(40), *XOR, backprop_step=3, backprop_case="C")
    assert "backprop" in s.views and s.backprop_step == 3 and s.backprop_case == 2
    assert any(ax.get_title(loc="left").startswith("backward pass of C") for ax in s.fig.axes)
    with pytest.raises(ValueError, match="backprop_step must be a whole number from 0 to 6"):
        explore(reference_net(40), *XOR, backprop_step=7)
    with pytest.raises(ValueError, match="is not a group"):
        explore(reference_net(40), *XOR, backprop_case="Z")


def test_the_derivation_is_the_widest_lower_panel(static):
    for style, share in (("wide", 0.3), ("standard", 1.0)):
        s = explore(reference_net(40), *XOR, views="backprop", layout=style, backprop_step=1)
        ax = [a for a in s.fig.axes if a.get_title(loc="left").startswith("backward pass of A")][0]
        others = [a for a in s.fig.axes[4:] if a is not ax]
        assert width_share(ax) >= share and all(width_share(a) < width_share(ax) for a in others)
        plt.close("all")


def test_record_backprop(tmp_path):
    from PIL import Image

    s = Session(reference_net(40), *XOR)
    w = s.weights()
    path = s.record_backprop(tmp_path / "bp.gif", dpi=30)
    with Image.open(path) as im:
        assert im.n_frames == 7
    assert s.weights() == w and s.views == () and s.backprop_step == 0
    assert plt.get_fignums() == []


def test_widget_controls(monkeypatch):
    pytest.importorskip("ipywidgets")
    import IPython.display

    monkeypatch.delenv("NB_STATIC", raising=False)
    monkeypatch.setattr(IPython.display, "display", lambda *a, **k: None)
    s = explore(reference_net(40), *XOR, views="backprop")
    C = s.controller
    assert C.tabs.selected_index == 1 and "backprop" in s.views and "Press next stage" in C.message.value
    png = C.image.value
    C.bp_next.click()
    C.bp_next.click()
    assert s.backprop_step == 2 and C.image.value != png
    assert "On the purple lines" in C.message.value
    assert not C.bp_prev.disabled and not C.bp_next.disabled
    C.bp_prev.click()
    assert s.backprop_step == 1
    C.bp_case.value = "D"
    assert s.backprop_case == 3 and any(ax.get_title(loc="left").startswith("backward pass of D") for ax in s.fig.axes)
    png = C.image.value
    C.tabs.selected_index = 0  # the other tab: the backward pass stays in the picture, which does not change
    assert "backprop" in s.views and s.backprop_step == 1 and C.image.value == png
    C.bp_step.value = 0
    assert C.bp_prev.disabled and not C.bp_next.disabled  # the stage buttons stop at the ends
    C.bp_step.value = 6
    assert C.bp_next.disabled and not C.bp_prev.disabled
    starts_on = explore(reference_net(40), *XOR, backprop_step=3).controller
    assert starts_on.tabs.selected_index == 1
    plt.close("all")


def test_shared_labels_only_in_the_first_panel(static):
    """Weights (forward, back to h) are the same in every panel: panel A only. Per-case slopes: every panel."""
    s = explore(reference_net(40), *XOR, views="backprop")
    count = lambda fig: [sum(t.get_fontsize() == THEME.edge_label_size for t in ax.texts) for ax in network_panels(fig)]  # noqa: E731
    for k, expected in ((0, [13, 0, 0, 0]), (2, [4, 4, 4, 4]), (3, [3, 0, 0, 0]), (6, [13, 13, 13, 13])):
        s.backprop_step = k
        assert count(s.draw()) == expected, k
        plt.close("all")
